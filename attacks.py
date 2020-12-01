from foolbox import attacks as fa
from foolbox.distances import LpDistance

class attack_orth(fa.base.MinimizationAttack):
    def __init__(self,input_attack,params, adv_dirs=[],orth_const=50):
        super(attack_orth,self).__init__()
        self.input_attack = input_attack(**params)
        self.distance = LpDistance(2)
        self.dirs = adv_dirs
        self.orth_const = orth_const


    def run(self,model,inputs,criterion,**kwargs):
        return self.input_attack.run(model,inputs,criterion,dirs=self.dirs,orth_const=self.orth_const, **kwargs)
    def distance(self):
        ...


class CarliniWagner(fa.L2CarliniWagnerAttack):
    def run

# class L2CarliniWagnerAttack(MinimizationAttack):
#     fa.L2CarliniWagnerAttack
#     """Implementation of the Carlini & Wagner L2 Attack. [#Carl16]_
#
#     Args:
#         binary_search_steps : Number of steps to perform in the binary search
#             over the const c.
#         steps : Number of optimization steps within each binary search step.
#         stepsize : Stepsize to update the examples.
#         confidence : Confidence required for an example to be marked as adversarial.
#             Controls the gap between example and decision boundary.
#         initial_const : Initial value of the const c with which the binary search starts.
#         abort_early : Stop inner search as soons as an adversarial example has been found.
#             Does not affect the binary search over the const c.
#
#     References:
#         .. [#Carl16] Nicholas Carlini, David Wagner, "Towards evaluating the robustness of
#             neural networks. In 2017 ieee symposium on security and privacy"
#             https://arxiv.org/abs/1608.04644
#     """
#
#     distance = l2
#
#     def __init__(
#         self,
#         binary_search_steps: int = 9,
#         steps: int = 10000,
#         stepsize: float = 1e-2,
#         confidence: float = 0,
#         initial_const: float = 1e-3,
#         abort_early: bool = True,
#     ):
#         self.binary_search_steps = binary_search_steps
#         self.steps = steps
#         self.stepsize = stepsize
#         self.confidence = confidence
#         self.initial_const = initial_const
#         self.abort_early = abort_early
#
#     def run(
#         self,
#         model: Model,
#         inputs: T,
#         criterion: Union[Misclassification, TargetedMisclassification, T],
#         *,
#         early_stop: Optional[float] = None,
#         proj_matrix: Any,
#         orth_const: float = None,
#         **kwargs: Any,
#     ) -> T:
#         raise_if_kwargs(kwargs)
#         x, restore_type = ep.astensor_(inputs)
#         criterion_ = get_criterion(criterion)
#         del inputs, criterion, kwargs
#
#         if orth_const != None:
#             A = ep.astensor(proj_matrix)
#
#         N = len(x)
#
#         if isinstance(criterion_, Misclassification):
#             targeted = False
#             classes = criterion_.labels
#             change_classes_logits = self.confidence
#         elif isinstance(criterion_, TargetedMisclassification):
#             targeted = True
#             classes = criterion_.target_classes
#             change_classes_logits = -self.confidence
#         else:
#             raise ValueError("unsupported criterion")
#
#         def is_adversarial(perturbed: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
#             if change_classes_logits != 0:
#                 logits += ep.onehot_like(logits, classes, value=change_classes_logits)
#             return criterion_(perturbed, logits)
#
#         if classes.shape != (N,):
#             name = "target_classes" if targeted else "labels"
#             raise ValueError(
#                 f"expected {name} to have shape ({N},), got {classes.shape}"
#             )
#
#         bounds = model.bounds
#         to_attack_space = partial(_to_attack_space, bounds=bounds)
#         to_model_space = partial(_to_model_space, bounds=bounds)
#
#         x_attack = to_attack_space(x)
#         reconstsructed_x = to_model_space(x_attack)
#
#         rows = range(N)
#
#         def loss_fun(
#                 delta: ep.Tensor, consts: ep.Tensor
#         ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
#             assert delta.shape == x_attack.shape
#             assert consts.shape == (N,)
#
#             x = to_model_space(x_attack + delta)
#             logits = model(x)
#
#             if targeted:
#                 c_minimize = best_other_classes(logits, classes)
#                 c_maximize = classes  # target_classes
#             else:
#                 c_minimize = classes  # labels
#                 c_maximize = best_other_classes(logits, classes)
#
#             is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
#             assert is_adv_loss.shape == (N,)
#
#             is_adv_loss = is_adv_loss + self.confidence
#             is_adv_loss = ep.maximum(0, is_adv_loss)
#             is_adv_loss = is_adv_loss * consts
#             if orth_const != None:
#                 print('it works!')
#                 dir =  x.flatten()
#                 dir_norm = dir.square().sum(axis=-1).sqrt()
#                 is_adv_loss += orth_const * (A - dir_norm.reshape([len(dir_norm),1]).matmul(dir_norm.reshape([1,len(dir_norm)])))
#
#             squared_norms = flatten(x - reconstsructed_x).square().sum(axis=-1)
#             loss = is_adv_loss.sum() + squared_norms.sum()
#             return loss, (x, logits)
#
#         loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)
#
#         consts = self.initial_const * np.ones((N,))
#         lower_bounds = np.zeros((N,))
#         upper_bounds = np.inf * np.ones((N,))
#
#         best_advs = ep.zeros_like(x)
#         best_advs_norms = ep.full(x, (N,), ep.inf)
#
#         # the binary search searches for the smallest consts that produce adversarials
#         for binary_search_step in range(self.binary_search_steps):
#             if (
#                 binary_search_step == self.binary_search_steps - 1
#                 and self.binary_search_steps >= 10
#             ):
#                 # in the last binary search step, repeat the search once
#                 consts = np.minimum(upper_bounds, 1e10)
#
#             # create a new optimizer find the delta that minimizes the loss
#             delta = ep.zeros_like(x_attack)
#             optimizer = AdamOptimizer(delta)
#
#             # tracks whether adv with the current consts was found
#             found_advs = np.full((N,), fill_value=False)
#             loss_at_previous_check = np.inf
#
#             consts_ = ep.from_numpy(x, consts.astype(np.float32))
#
#             for step in range(self.steps):
#                 loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_)
#                 delta += optimizer(gradient, self.stepsize)
#
#                 if self.abort_early and step % (np.ceil(self.steps / 10)) == 0:
#                     # after each tenth of the overall steps, check progress
#                     if not (loss <= 0.9999 * loss_at_previous_check):
#                         break  # stop Adam if there has been no progress
#                     loss_at_previous_check = loss
#
#                 found_advs_iter = is_adversarial(perturbed, logits)
#                 found_advs = np.logical_or(found_advs, found_advs_iter.numpy())
#
#                 norms = flatten(perturbed - x).norms.l2(axis=-1)
#                 closer = norms < best_advs_norms
#                 new_best = ep.logical_and(closer, found_advs_iter)
#
#                 new_best_ = atleast_kd(new_best, best_advs.ndim)
#                 best_advs = ep.where(new_best_, perturbed, best_advs)
#                 best_advs_norms = ep.where(new_best, norms, best_advs_norms)
#
#             upper_bounds = np.where(found_advs, consts, upper_bounds)
#             lower_bounds = np.where(found_advs, lower_bounds, consts)
#
#             consts_exponential_search = consts * 10
#             consts_binary_search = (lower_bounds + upper_bounds) / 2
#             consts = np.where(
#                 np.isinf(upper_bounds), consts_exponential_search, consts_binary_search
#             )
#
#         return restore_type(best_advs)
#
