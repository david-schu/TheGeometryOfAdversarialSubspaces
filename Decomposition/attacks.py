from typing import Union, Tuple, Any, Optional
from functools import partial
import numpy as np
from scipy.optimize import minimize, LinearConstraint
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
            # adv = (z.reshape((1,-1)).matmul(basis_)).reshape(x.shape) + x

            adv = delta + x
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

        def loss_and_grad(delta, consts):
            delta = ep.from_numpy(x, delta.astype(np.float32)).reshape(x.shape)
            loss, _, gradient = loss_aux_and_grad(delta, consts)
            loss_np = loss.numpy()
            grad_np = gradient.flatten().numpy()
            return loss_np, grad_np

        x_np = x.flatten().numpy()

        # basis = make_orth_basis(dirs)/1e2
        # basis_ = ep.from_numpy(x, basis.astype(np.float32))
        #
        # cons = LinearConstraint(basis.T, lb=-x_np, ub=1-x_np)
        #
        # z = np.zeros(len(basis))

        bnds = np.vstack((-x_np, 1-x_np)).T#np.repeat([[0, 1]], 784, axis=0)
        cons = ()
        if len(dirs)>0:
            for d in dirs:
                con = LinearConstraint(d, lb=0, ub=0)
                cons = cons + (con,)

        best_bin_search_step = 0
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

            consts_ = ep.from_numpy(x, consts.astype(np.float32))

            res = minimize(loss_and_grad, np.zeros(len(x_np)), jac=True, bounds=bnds, args=(consts_),
                           method='trust-constr', constraints=cons, options={'maxiter': self.steps, 'verbose':2})

            perturbed = ep.from_numpy(x, res.x.astype(np.float32)+x_np).reshape(x.shape)
            valid_res = True

            if perturbed.max() > 1.001 or perturbed.min() < -0.001:
                valid_res = False
            elif ep.all(perturbed==0):
                valid_res = False
            else:
                perturbed = perturbed.clip(0, 1)

            # tracks whether adv with the current consts was found
            found_advs = np.full((N,), fill_value=False)
            if valid_res:
                logits = model(perturbed)

                found_advs_iter = is_adversarial(perturbed, logits)
                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())

                norms = (perturbed - x).flatten().norms.l2(axis=-1)
                closer = norms < best_advs_norms
                new_best = ep.logical_and(closer, found_advs_iter)

                if new_best:
                    best_bin_search_step=consts

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
        print('Binary search step: ' + str(best_bin_search_step))
        return restore_type(best_advs)

def make_orth_basis(dirs):
    n_iterations = 3
    n_pixel = 784#dirs.shape[-1]
    basis = np.random.uniform(-1, 1, (n_pixel - len(dirs), n_pixel))
    basis = basis / np.linalg.norm(basis, axis=-1, keepdims=True)
    if len(dirs)>0:
        basis_with_dirs = np.concatenate((dirs, basis), axis=0)
    else:
        basis_with_dirs = basis

    for it in range(n_iterations):
        for i, v in enumerate(basis):
            v_orth = v - ((basis_with_dirs[:len(dirs) + i] * v.reshape((1, -1))).sum(-1, keepdims=True) *
                          basis_with_dirs[:len(dirs) + i]).sum(0)
            u_orth = v_orth / np.linalg.norm(v_orth)
            basis_with_dirs[len(dirs)+i] = u_orth
            basis[i] = u_orth

    return basis