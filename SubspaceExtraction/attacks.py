from typing import Union, Tuple, Any, Optional
import numpy as np
import eagerpy as ep
import foolbox as fb
from foolbox import attacks as fa
from foolbox.models import Model
from foolbox.distances import LpDistance
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.base import MinimizationAttack, T, get_criterion, raise_if_kwargs

from cyipopt import minimize_ipopt


class OrthogonalAttack(MinimizationAttack):
    def __init__(self, input_attack, params, adv_dirs=[], random_start=False):
        super(OrthogonalAttack, self).__init__()
        self.input_attack = input_attack(**params)
        self.distance = LpDistance(2)
        self.dirs = adv_dirs
        self.random_start = random_start

    def run(self, model, inputs, criterion, **kwargs):
        return self.input_attack.run(model, inputs, criterion, dirs=self.dirs, random_start=self.random_start, **kwargs)

    def distance(self):
        ...


class L2OrthAttack(fa.L2CarliniWagnerAttack):
    def run(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Misclassification, TargetedMisclassification, T],
            *,
            early_stop: Optional[float] = None,
            random_start: Optional[float] = None,
            dirs: Optional[Any] = [],
            **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        N = len(x)

        labels = criterion_.labels
        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
            change_classes_logits = self.confidence
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
            change_classes_logits = -self.confidence
        else:
            raise ValueError("unsupported criterion")

        def is_adversarial(perturbed: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
            if change_classes_logits != 0:
                logits += ep.onehot_like(logits, classes, value=change_classes_logits)
            return criterion_(perturbed, logits)

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        rows = range(N)

        def loss_fun(
                adv: ep.Tensor, consts: ep.Tensor
        ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:

            logits = model(adv)

            if targeted:
                c_minimize = fa.carlini_wagner.best_other_classes(logits, classes)
                c_maximize = classes  # target_classes
                is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            else:

                is_adv_loss = 1 / (ep.crossentropy(logits, labels)+1e-15)

            assert is_adv_loss.shape == (N,)

            is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts

            squared_norms = (adv - x).flatten(1, -1).square().sum(axis=-1)

            loss = is_adv_loss.sum() + squared_norms.sum()

            return loss, (adv, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        def loss_and_grad(var_opt, consts):
            var_opt = ep.from_numpy(x, var_opt.astype(np.float64)).reshape(x.shape)
            loss, _, gradient = loss_aux_and_grad(var_opt, consts)
            loss_np = loss.numpy().item()
            grad_np = gradient.flatten().numpy()
            return loss_np, grad_np

        x_np = x.flatten().numpy()

        bnds = [(0, 1) for _ in range(len(x_np))]

        cons = ()

        if len(dirs)>0:
            for d in dirs:
                con = {'type': 'eq', 'fun': lambda adv, d, x_np: ((adv-x_np)*d).sum(), 'args': (d, x_np, ),
                       'jac': lambda adv, d, x_np: d}
                cons = cons + (con,)

        consts = self.initial_const * np.ones((N,))
        lower_bounds = np.zeros((N,))
        upper_bounds = np.inf * np.ones((N,))

        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.full(x, (N,), ep.inf)
        # the binary search searches for the smallest consts that produce adversarials
        count = 0
        for binary_search_step in range(self.binary_search_steps):
            if (
                    binary_search_step == self.binary_search_steps - 1
                    and self.binary_search_steps >= 10
            ):
                # in the last binary search step, repeat the search once
                consts = np.minimum(upper_bounds, 1e10)

            consts_ = ep.from_numpy(x, consts.astype(np.float64))

            init = (x_np+np.random.normal(scale=1, size=x_np.shape)).clip(0, 1)
            res = minimize_ipopt(loss_and_grad, x0=init,
                                 jac=True, constraints=cons, args=(consts_),  bounds=bnds,
                                 options={'maxiter': self.steps, 'disp': 0, 'jac_c_constant': 'yes',
                                           'jac_d_constant': 'yes'})

            perturbed = ep.from_numpy(x, (res.x).astype(np.float64)).reshape(x.shape)


            valid_res = True

            if perturbed.max() > 1.001 or perturbed.min() < -0.001:
                valid_res = False
            elif ep.all(perturbed == 0):
                valid_res = False
            else:
                perturbed = perturbed.clip(0, 1)

            # tracks whether adv with the current consts was found
            found_advs = np.full((N,), fill_value=False)
            if valid_res:
                logits = model(perturbed)

                found_advs_iter = is_adversarial(perturbed, logits)
                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())

                if found_advs_iter:
                    count += 1
                    pert = perturbed - x
                    n_scales = 1000
                    batchsize = 100
                    scaled_pert = ep.from_numpy(x, np.linspace(.5, 1, n_scales).astype(np.float64)).reshape((-1, 1, 1, 1)) \
                                  * pert.reshape(x.shape[1:])
                    md_in = x + scaled_pert
                    correct_classes = np.zeros(n_scales)
                    for batch in range(int(n_scales/batchsize)):
                        correct_classes[batch*batchsize:(batch+1)*batchsize] = \
                            (model(md_in[batch*batchsize:(batch+1)*batchsize]).argmax(axis=1) == labels).numpy()
                    idx = int(correct_classes.sum())
                    perturbed = x + scaled_pert[idx].reshape(x.shape)

                norms = (perturbed - x).flatten().norms.l2(axis=-1)
                closer = norms < best_advs_norms
                new_best = ep.logical_and(closer, found_advs_iter)

                new_best_ = fb.devutils.atleast_kd(new_best, best_advs.ndim)
                best_advs = ep.where(new_best_, perturbed, best_advs)
                best_advs_norms = ep.where(new_best, norms, best_advs_norms)

            upper_bounds = np.where(found_advs, consts, upper_bounds)
            lower_bounds = np.where(found_advs, lower_bounds, consts)

            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = np.where(
                np.isinf(upper_bounds), consts_exponential_search, consts_binary_search
            )
            if count == 3:
                break

        return restore_type(best_advs)
