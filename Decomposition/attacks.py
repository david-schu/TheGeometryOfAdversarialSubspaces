from typing import Union, Tuple, Any, Optional
from functools import partial
import numpy as np
from scipy.optimize import minimize
import eagerpy as ep
import foolbox as fb
from foolbox import attacks as fa
from foolbox.models import Model
from foolbox.distances import LpDistance
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.base import MinimizationAttack, T, get_criterion, raise_if_kwargs

import torch
from utils import dev

class OrthogonalAttack(MinimizationAttack):
    def __init__(self, input_attack, params, adv_dirs=[], random_start=False):
        super(OrthogonalAttack,self).__init__()
        self.input_attack = input_attack(**params)
        self.distance = LpDistance(2)
        self.dirs = adv_dirs
        self.random_start = random_start

    def run(self, model, inputs, criterion, **kwargs):
        return self.input_attack.run(model, inputs, criterion, dirs=self.dirs, random_start=self.random_start, **kwargs)

    def distance(self):
        ...


class CarliniWagner(fa.L2CarliniWagnerAttack):
    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        early_stop: Optional[float] = None,
        random_start: Optional[float] = None,
        dirs: Optional[Any] = [],
        ** kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        N = len(x)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
            change_classes_logits = self.confidence
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
            change_classes_logits = -self.confidence
        else:
            raise ValueError("unsupported 500criterion")

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
                delta: ep.Tensor, consts: ep.Tensor
        ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
            assert delta.shape == x.shape
            assert consts.shape == (N,)

            adv = delta

            logits = model(adv)

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
            squared_norms = (adv - x).flatten(1,-1).square().sum(axis=-1)

            loss = is_adv_loss.sum() + squared_norms.sum()

            return loss, (adv, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        def loss_and_grad(adv, consts):
            adv_ = ep.from_numpy(x, adv.astype(np.float32)).reshape(x.shape)
            loss, _, gradient = loss_aux_and_grad(adv_, consts)
            loss_np = loss.numpy()
            grad_np = gradient.flatten().numpy()
            return loss_np, grad_np

        x_np = x.flatten().numpy()
        bnds = np.repeat([[0, 1]], 784, axis=0)

        cons = ()
        if len(dirs) > 0:
            for d in dirs:
                con = {'type': 'eq', 'fun': lambda adv, d, x_np: ((adv-x_np)*d).sum(), 'args': (d, x_np, )}
                cons = cons + (con,)

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
            delta = ep.zeros_like(x)
            if random_start:
                delta = (delta.uniform(shape=delta.shape)-x)*1e-2

            consts_ = ep.from_numpy(x, consts.astype(np.float32))

            res = minimize(loss_and_grad, (x+delta).flatten().numpy(), jac=True, args=(consts_), method='SLSQP',
                           constraints=cons, bounds=bnds, options={'maxiter': 500, 'disp': False, "iprint": 0})
            # print(res.message)

            perturbed = ep.from_numpy(x, res.x.astype(np.float32)).reshape(x.shape)
            logits = model(perturbed)


            # tracks whether adv with the current consts was found
            found_advs = np.full((N,), fill_value=False)

            found_advs_iter = is_adversarial(perturbed, logits)
            found_advs = np.logical_or(found_advs, found_advs_iter.numpy())

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

        return restore_type(best_advs)

