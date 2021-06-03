from typing import Union, Tuple, Any, Optional
from functools import partial
import numpy as np
import torch
import eagerpy as ep
import foolbox as fb
from foolbox import attacks as fa
from foolbox.models import Model
from foolbox.distances import LpDistance
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.base import MinimizationAttack, T, get_criterion, raise_if_kwargs


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
        dirs = ep.astensor(dirs)  ##################
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
        # if len(dirs>0):
        #     dirs = ep.astensor(dirs)
        rows = range(N)

        def loss_fun(
                delta: ep.Tensor, consts: ep.Tensor
        ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
            assert delta.shape == x_attack.shape
            assert consts.shape == (N,)

            ######## by David ############
            adv = to_model_space(x_attack + delta)
            if len(dirs) > 0:
                # orth_loss = False

            # else:
                # orth_loss = False
                s = adv - reconstructed_x
                # gram_schmidt = torch.zeros(delta.shape)
                # for i in range(len(delta)):
                #     dirs_i = dirs[i].flatten(-2, -1)
                #     s_dir = s.float32()[i].flatten(-2, -1).flatten(-2, -1).expand_dims(-1)
                #     gram_schmidt[i] = (dirs_i.matmul(s_dir)*dirs_i).sum(0).reshape(delta.shape).raw
                # adv = (adv-gram_schmidt)
                # if adv.max()>1 or adv.min()<0:
                #     orth_loss = True

                for i in range(len(delta)):
                    x_i = reconstructed_x[i].flatten()
                    d_i = dirs[i].flatten(1, -1)
                    s_i = s.float32()[i].flatten()

                    gram_schmidt = (d_i.matmul(s_i.expand_dims(-1))*d_i).sum(0)
                    s_scaled = torch.linspace(0, 1, 100).outer((s_i - gram_schmidt).raw)
                    a_scaled = x_i - s_scaled

                    larger = (a_scaled<=1).any(1)
                    smaller = (a_scaled>=0).any(1)
                    out_of_range = larger.logical_and(smaller).raw
                    idx = out_of_range.nonzero(as_tuple=False)[-1]

                    adv.raw[i] = (a_scaled[idx]).reshape(adv[i].shape).raw

            logits = model(adv)
            # ###############################

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
            squared_norms = (adv - reconstructed_x).flatten().square().sum(axis=-1)

            # if orth_loss:
            #     is_orth = dirs * (adv-reconstructed_x).flatten(-2, -1)
            #     is_orth = is_orth.sum(axis=-1).square().sum(axis=-1)
            #     loss = is_adv_loss.sum() + squared_norms.sum() + 1e6 * is_orth.sum()
            # else:
            loss = is_adv_loss.sum() + squared_norms.sum()

            return loss, (adv, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

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
            if random_start:
                delta = delta.uniform(shape=delta.shape, low=0, high=0.1)

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
