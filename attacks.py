from plots import plot_losses

from typing import Union, Tuple, Any, Optional
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import eagerpy as ep
import foolbox as fb
from foolbox import attacks as fa
from foolbox import distances, models, criteria


class OrthogonalAttack(fa.base.MinimizationAttack):
    def __init__(self,input_attack,params, adv_dirs=[],orth_const=50):
        super(OrthogonalAttack,self).__init__()
        self.input_attack = input_attack(**params)
        self.distance = distances.LpDistance(2)
        self.dirs = adv_dirs
        self.orth_const = orth_const

    def run(self,model,inputs,criterion,**kwargs):
        return self.input_attack.run(model,inputs,criterion,dirs=self.dirs,orth_const=self.orth_const, **kwargs)

    def distance(self):
        ...


class CarliniWagner(fa.L2CarliniWagnerAttack):
    def run(
        self,
        model: models.Model,
        inputs: fa.base.T,
        criterion: Union[criteria.Misclassification, criteria.TargetedMisclassification, fa.base.T],
        *,
        early_stop: Optional[float] = None,
        dirs: Optional[Any] = [],
        orth_const: Optional[float] = 50,
        ** kwargs: Any,
    ) -> fa.base.T:
        fa.base.raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = fa.base.get_criterion(criterion)
        dirs = ep.astensor(dirs)  ##################
        del inputs, criterion, kwargs

        N = len(x)

        if isinstance(criterion_, criteria.Misclassification):
            targeted = False
            classes = criterion_.labels
            change_classes_logits = self.confidence
        elif isinstance(criterion_, criteria.TargetedMisclassification):
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

        bounds = model.bounds
        to_attack_space = partial(fa.carlini_wagner._to_attack_space, bounds=bounds)
        to_model_space = partial(fa.carlini_wagner._to_model_space, bounds=bounds)

        x_attack = to_attack_space(x)
        reconstructed_x = to_model_space(x_attack)

        rows = range(N)

        def loss_fun(
                delta: ep.Tensor, consts: ep.Tensor
        ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
            assert delta.shape == x_attack.shape
            assert consts.shape == (N,)

            adv = to_model_space(x_attack + delta)

            ######## by David ############

            if (len(dirs) == 0):
                logits = model(adv)
                is_orth = ep.zeros(delta, 1)
            else:
                new_dir = (adv - reconstructed_x).reshape([dirs.shape[0], 1, dirs.shape[-1]])
                is_orth = (new_dir * dirs).sum(axis=-1)
                is_orth = is_orth.square().sum(axis=-1)
                is_orth = is_orth * orth_const

                s = adv - reconstructed_x
                gram_schmidt = ep.zeros_like(s)
                for i, a in enumerate(adv):
                    gram_schmidt.raw[i] = (dirs.float32()[i].matmul(s[i].reshape([dirs.shape[-1], 1])) * dirs[i]).sum(
                        axis=0).reshape(s[i].raw.shape).raw
                adv_orth = adv - gram_schmidt
                logits = model(adv_orth)
                # orth_const = np.maximum(min_orth,min_orth * gram_schmidt.abs().sum().item())
                adv = ep.zeros_like(x_attack)
                adv += adv_orth.raw
            ###############################

            if targeted:
                c_minimize = fa.carlini_wagner.best_other_classes(logits, classes)
                c_maximize = classes  # target_classes
            else:
                c_minimize = classes  # labels
                c_maximize = fa.carlini_wagner.best_other_classes(logits, classes)

            is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)

            is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts
            squared_norms = fb.devutils.flatten(adv - reconstructed_x).square().sum(axis=-1)

            ######## by David ############
            loss = is_adv_loss.sum() + squared_norms.sum() + is_orth.sum()
            losses[binary_search_step, :, step] = np.array(
                [loss.item(), is_adv_loss.sum().item(), squared_norms.sum().item(), is_orth.sum().item()])
            ###############################

            return loss, (adv, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        losses = np.zeros([self.binary_search_steps, 4, self.steps])
        best_binary_search_step = 0

        consts = self.initial_const * np.ones((N,))
        lower_bounds = np.zeros((N,))
        upper_bounds = np.inf * np.ones((N,))

        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.full(x, (N,), ep.inf)

        # the binary search searches for the smallest consts that produce adversarials
        for binary_search_step in range(self.binary_search_steps):
            if (
                    binary_search_step == self.binary_search_steps - 1
                    and self.binary_search_steps >= 10
            ):
                # in the last binary search step, repeat the search once
                consts = np.minimum(upper_bounds, 1e10)

            # create a new optimizer find the delta that minimizes the loss
            delta = ep.zeros_like(x_attack)
            optimizer = fa.carlini_wagner.AdamOptimizer(delta)

            # tracks whether adv with the current consts was found
            found_advs = np.full((N,), fill_value=False)
            loss_at_previous_check = np.inf

            consts_ = ep.from_numpy(x, consts.astype(np.float32))
            for step in range(self.steps):

                loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_)
                delta += optimizer(gradient, self.stepsize)

                if self.abort_early and step % (np.ceil(self.steps / 10)) == 0:
                    # after each tenth of the overall steps, check progress
                    if not (loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has been no progress
                    loss_at_previous_check = loss

                found_advs_iter = is_adversarial(perturbed, logits)
                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())

                norms = fb.devutils.flatten(perturbed - x).norms.l2(axis=-1)
                closer = norms < best_advs_norms
                new_best = ep.logical_and(closer, found_advs_iter)

                new_best_ = fb.devutils.atleast_kd(new_best, best_advs.ndim)
                best_advs = ep.where(new_best_, perturbed, best_advs)
                best_advs_norms = ep.where(new_best, norms, best_advs_norms)

                # if new_best:
                #     best_binary_search_step = binary_search_step

            upper_bounds = np.where(found_advs, consts, upper_bounds)
            lower_bounds = np.where(found_advs, lower_bounds, consts)

            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = np.where(
                np.isinf(upper_bounds), consts_exponential_search, consts_binary_search
            )
        if len(dirs):
            fig, ax = plot_losses(losses[best_binary_search_step])
            plt.suptitle('Loss functions for orth_const = ' + str(orth_const))
            plt.show()

        return restore_type(best_advs)
