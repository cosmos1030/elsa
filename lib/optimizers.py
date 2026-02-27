import torch
from typing import Union, Dict
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate
import math

# Import base optimizers
from torch.optim import Adam, AdamW, SGD
from torch.optim.optimizer import _get_scalar_dtype, _device_dtype_check_for_fused
from torchao.optim import Adam8bit,Adam4bit
from .utils import FP8Config, FP8State, ScalingType

def _is_dtensor(x): 
    return hasattr(x, "to_local")

def _loc(x):
    # Return local shard if DTensor, otherwise the tensor itself
    return x.to_local() if _is_dtensor(x) else x

def get_admm_optimizer(base_optimizer_cls):
    """
    Factory function to create an ADMM optimizer class that inherits from a base optimizer.
    This preserves the single-class structure required for FSDP compatibility.
    """
    base_optimizer_cls = base_optimizer_cls.lower()
    if base_optimizer_cls not in ['adam', 'adamw', 'adam8bit', 'adam4bit', 'sgd']:
        raise ValueError("base_optimizer_cls must be one of 'adam', 'adamw', 'adam8bit', 'adam4bit', or 'sgd'.")
    if base_optimizer_cls == 'adam':
        base_optimizer_cls = Adam
    elif base_optimizer_cls == 'adamw':
        base_optimizer_cls = AdamW
    elif base_optimizer_cls == 'adam8bit':
        base_optimizer_cls = Adam8bit
    elif base_optimizer_cls == 'adam4bit':
        base_optimizer_cls = Adam4bit
    elif base_optimizer_cls == 'sgd':
        base_optimizer_cls = SGD
    class ADMMOptimizer(base_optimizer_cls):
        """
        ADMM optimizer built by subclassing a base optimizer (e.g., Adam).
        - Proximal term is added AFTER gradient clipping and BEFORE the actual step.
        - Compatible with FSDP2/DTensor: all state kept per-shard, reductions in fp32.
        """
        def __init__(
            self,
            param_groups,
            projection_fn,
            sparsity: float,
            interval: int,
            # ADMM specific arguments
            lmda: float = 1e-3, # For constant schedule
            init_lmda: float = 0.0, # For scheduling
            final_lmda: float = 0.01, # For scheduling
            lmda_schedule_mode: str = 'constant', # 'constant', 'linear', 'cosine', 'exponential'
            total_steps: int = 1, # Total steps for fixed lmda schedules
            prune_n: int = 0,
            prune_m: int = 0,
            projection_mode: str = "identity",   # 'identity' | 'momentum'
            projection_bias_correction: bool = False, # use bias correction in projection (for momentum)
            dual_dtype: str = 'fp32',
            split_dtype: str = 'fp32',
            accelerator=None,                    # optional: to get world_size and device
            init_lambda_from_inv_resid: bool = False,
            **base_optimizer_kwargs
        ):
            super().__init__(param_groups, **base_optimizer_kwargs)

            # --- ADMM config ---
            self.projection      = projection_fn
            self.sparsity        = float(sparsity)
            self.interval        = int(interval)
            self.init_lmda = float(init_lmda)
            self.final_lmda = float(final_lmda)
            self.lmda_schedule_mode = lmda_schedule_mode.lower()
            self.init_lambda_from_inv_resid = init_lambda_from_inv_resid

            if self.lmda_schedule_mode == 'constant':
                self.lmda_default = float(lmda)
            else:
                self.lmda_default = float(init_lmda)

            self.total_steps     = int(total_steps)
            self.prune_n         = int(prune_n)
            self.prune_m         = int(prune_m)
            self.projection_mode  = projection_mode.lower()
            self.projection_bias_correction = bool(projection_bias_correction)

            if self.lmda_schedule_mode != 'constant' and self.init_lmda is None:
                raise ValueError("For lambda scheduling, init_lmda must be provided.")

            if dual_dtype == 'bf16':
                self.dual_dtype = torch.bfloat16
            elif dual_dtype == 'fp32':
                self.dual_dtype = torch.float32
            elif dual_dtype == 'float8_e5m2':
                self.dual_dtype = torch.float8_e5m2
            elif dual_dtype == 'float8_e4m3fn':
                self.dual_dtype = torch.float8_e4m3fn
            else:
                raise ValueError(f"Unsupported dual_dtype: {dual_dtype}")

            if split_dtype == 'bf16':
                self.split_dtype = torch.bfloat16
            elif split_dtype == 'fp32':
                self.split_dtype = torch.float32
            elif split_dtype == 'float8_e5m2':
                self.split_dtype = torch.float8_e5m2
            elif split_dtype == 'float8_e4m3fn':
                self.split_dtype = torch.float8_e4m3fn
            else:
                raise ValueError(f"Unsupported split_dtype: {split_dtype}")

            if self.projection_mode not in ("identity", "momentum"):
                raise ValueError(f"projection_mode must be 'identity' or 'momentum', got {self.projection_mode}")
            if self.lmda_schedule_mode not in ('constant', 'linear', 'cosine', 'exponential'):
                raise ValueError(f"lmda_schedule_mode must be 'constant', 'linear', 'cosine', or 'exponential', got {self.lmda_schedule_mode}")

            # Runtime helpers
            self.accelerator = accelerator
            self.process_group = getattr(accelerator, "process_group", None) if accelerator is not None else None
            self.current_step = 0
            self.mask_metrics = {'step_hamming': 0.0, 'initial_hamming': 0.0, 'step_iou': 0.0, 'initial_iou': 0.0}

        def _lazy_init_admm_state(self, p: torch.nn.Parameter, group: Dict):
            """
            Lazily initialize all required states for a parameter for both ADMM and the base optimizer.
            This must be called before the base optimizer's step if ADMM state is used before it,
            as it ensures the base optimizer's state is created before we add our own ADMM state.
            For Adam8bit support, make sure to pass group, gindx, pindx to initialize Adam8bit state properly.
            """
            st = self.state[p]
            if len(st) == 0: ## optimizer states for base optimizers 
                if isinstance(self, Adam): ## lazy init of Adam state, official implementation.
                    if group["fused"]:
                        _device_dtype_check_for_fused(p)
                    st["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    # Exponential moving average of gradient values
                    st["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    st["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        st["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                elif isinstance(self, (Adam4bit, Adam8bit)):
                    st["step"] = torch.tensor(0.0)
                    st["exp_avg"] = self._new_buffer(p, True)
                    st["exp_avg_sq"] = self._new_buffer(p, False)
                    if group["amsgrad"]:
                        st["max_exp_avg_sq"] = self._new_buffer(p, False)
                elif isinstance(self, SGD): ## sgd is stateless
                    pass
                else: 
                    raise NotImplementedError("Base optimizer state initialization not implemented for this optimizer.")
            if 'dual' in st: ## return if ADMM state is already initialized
                return

            # --- Initialize ADMM's state ---
            if self.dual_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                st["dual"] = FP8State.from_tensor(
                    torch.zeros_like(p), fp8_dtype=self.dual_dtype, granularity="tensorwise",
                    scaling_type=ScalingType.DYNAMIC, safety_margin=1.05,
                    sync_scales=True, process_group=self.process_group
                )
            else:
                st["dual"] = torch.zeros_like(p, dtype=self.dual_dtype, memory_format=torch.preserve_format)
            st["sparsity"] = self.sparsity


            init_importance = None
            # Initial split z and initial_split (as bool)
            z0 = self.projection([p.detach()], st["sparsity"], self.prune_n, self.prune_m,
                                 [init_importance], comparison_group="layer")[0]
            if self.init_lambda_from_inv_resid:
                initial_residual = torch.norm(p.detach() - z0.detach())
                st["lmda"] = self.lmda_default / (initial_residual.item() + 1e-8)
                st['prev_lmda'] = st["lmda"]
            else:
                st["lmda"] = self.lmda_default
                st['prev_lmda'] = self.lmda_default

            if self.split_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                st["split"] = FP8State.from_tensor(
                    z0, fp8_dtype=self.split_dtype, granularity="tensorwise",
                    scaling_type=ScalingType.DYNAMIC, safety_margin=1.05,
                    sync_scales=True, process_group=self.process_group
                )
            else:
                st["split"] = z0.detach().clone().to(device=p.device, dtype=self.split_dtype)
            st["initial_split"] = z0.detach().ne(0).clone().to(device=p.device)

        @torch.no_grad()
        def _proximal_update(self):
            """
            Add proximal term to gradients AFTER global gradient clipping and
            BEFORE the actual optimizer step. This ensures proximal is not clipped.
            We also scale proximal to match distributed gradient averaging.
            """
            # Determine world size for average scaling (DDP/FSDP usually average grads across ranks)
            if self.accelerator is not None and getattr(self.accelerator, "num_processes", None):
                world = int(self.accelerator.num_processes)
            elif dist.is_initialized():
                world = dist.get_world_size()
            else:
                world = 1
            avg_div = world if world > 0 else 1

            for g in self.param_groups:
                if not g.get("admm", False):
                    continue
                for w in g["params"]:
                    if w.grad is None:
                        continue
                    self._lazy_init_admm_state(w, g)
                    st = self.state[w]
                    dual, split = st["dual"], st["split"]
                    lmda = st["lmda"]
                    ## for fp8 states, upcast to fp32 for computation
                    dual = dual.dequant() if isinstance(dual, FP8State) else dual
                    split = split.dequant() if isinstance(split, FP8State) else split

                    # Proximal term: λ (w - z + u), add to gradient before optimizer step
                    penalty = w.detach() - split.detach() + dual.detach()
                    prox = lmda * penalty
                    prox_local = _loc(prox)
                    prox_local = prox_local.to(w.grad.dtype)
                    if avg_div > 1:
                        prox_local = prox_local / avg_div

                    if hasattr(w.grad, "to_local"):
                        gl = w.grad.to_local()
                        gl.add_(prox_local)
                    else:
                        w.grad.add_(prox_local)

        @torch.no_grad()
        def _dual_update(self):
            """
            Every 'interval' steps, update split (z) and dual (u), and compute mask_diff.
            - z^{k+1} = Proj(w + u)
            - u^{k+1} = u + α (w - z^{k+1})
            Also compute global mask flip ratio between old z and new z.
            """
            if (self.current_step % self.interval) != 0:
                return

            self.mask_metrics = {'step_hamming': 0.0, 'initial_hamming': 0.0, 'step_iou': 0.0, 'initial_iou': 0.0}
            admm_groups = 0

            for g in self.param_groups:
                if not g.get("admm", False):
                    continue
                admm_groups += 1
                weights = list(g["params"])
                if not weights:
                    continue

                device = weights[0].device
                flip_sum_step = torch.tensor(0, device=device, dtype=torch.int64)
                flip_sum_initial = torch.tensor(0, device=device, dtype=torch.int64)
                intersection_step = torch.tensor(0, device=device, dtype=torch.int64)
                union_step = torch.tensor(0, device=device, dtype=torch.int64)
                intersection_initial = torch.tensor(0, device=device, dtype=torch.int64)
                union_initial = torch.tensor(0, device=device, dtype=torch.int64)
                numel_sum = torch.tensor(0, device=device, dtype=torch.int64)

                for w in weights:
                    st = self.state[w]
                    initial_split = st["initial_split"]
                    spars = st["sparsity"]
                    current_lmda = st["lmda"]
                    previous_lmda = st["prev_lmda"]

                    ## for fp8 states, upcast to fp32 for computation
                    dual = st["dual"].dequant() if isinstance(st["dual"], FP8State) else st["dual"]
                    split = st["split"].dequant() if isinstance(st["split"], FP8State) else st["split"]

                    if current_lmda != previous_lmda:
                        dual.mul_(previous_lmda / current_lmda)

                    
                    importance_i = st.get("importance", None)
                    ## objective-aware projection
                    if self.projection_mode == "momentum":
                        v_t = st.get("exp_avg_sq",None)
                        if v_t is None:
                            raise ValueError("For momentum projection mode, optimizer must have 'exp_avg_sq' state (e.g., Adam).")
                        if self.projection_bias_correction:
                            beta2 = g.get('betas', (0.9, 0.95))[1]
                            importance_i = v_t / (1.0 - beta2**(st.get("step", 1)))
                        else:
                            importance_i = v_t
                        

                    z_in  = (w.detach() + dual.detach())
                    z_new = self.projection([z_in], spars, self.prune_n, self.prune_m,
                                            [importance_i], comparison_group="layer")[0]
                    z_new = z_new.detach().clone().to(w.device)

                    u_new = dual.detach() + (w.detach() - z_new)

                    w_l = _loc(w)
                    s_l = _loc(split)
                    d_l = _loc(dual)
                    z_new_l = _loc(z_new)

                    new_lmda_for_param = current_lmda
                    t = self.current_step
                    T = self.total_steps
                    s0 = self.init_lmda
                    s1 = self.final_lmda

                    if self.lmda_schedule_mode == 'constant':
                        new_lmda_for_param = current_lmda
                    elif self.lmda_schedule_mode == 'linear':
                        new_lmda_for_param = s0 + (s1 - s0) * (t / T)
                    elif self.lmda_schedule_mode == 'cosine':
                        new_lmda_for_param = s0 + (s1 - s0) * 0.5 * (1 - math.cos(math.pi * t / T))
                    elif self.lmda_schedule_mode == 'exponential':
                        if s1 <= 0:
                            raise ValueError("For exponential lambda schedule, final_lmda must be positive.")
                        if s0 < 0:
                            raise ValueError("For exponential lambda schedule, init_lmda must be non-negative.")

                        if s0 == 0:
                            s0_eff = 1e-12
                            new_lmda_for_param = s0_eff * (s1 / s0_eff)**(t / T)
                        else:
                            new_lmda_for_param = s0 * (s1/s0)**(t/T)

                    old_local = _loc(split)
                    new_local = _loc(z_new)
                    initial_local = _loc(initial_split)

                    old_mask = (old_local != 0)
                    new_mask = (new_local != 0)
                    initial_mask = initial_local

                    flip_local_step = (old_mask ^ new_mask).sum().to(device=device)
                    flip_local_initial = (initial_mask ^ new_mask).sum().to(device=device)
                    numel_local = torch.tensor(old_local.numel(), device=device)

                    intersection_step += (old_mask & new_mask).sum().to(device=device)
                    union_step += (old_mask | new_mask).sum().to(device=device)
                    intersection_initial += (initial_mask & new_mask).sum().to(device=device)
                    union_initial += (initial_mask | new_mask).sum().to(device=device)

                    flip_sum_step += flip_local_step
                    flip_sum_initial += flip_local_initial
                    numel_sum += numel_local

                    ## for fp8 states, requantize to save fp8 states
                    st['dual'].requant(u_new) if isinstance(st['dual'], FP8State) else dual.copy_(u_new)
                    st['split'].requant(z_new) if isinstance(st['split'], FP8State) else split.copy_(z_new)
                    
                    st["lmda"] = new_lmda_for_param
                    st["prev_lmda"] = current_lmda

                if dist.is_initialized():
                    dist.all_reduce(flip_sum_step,  op=dist.ReduceOp.SUM)
                    dist.all_reduce(flip_sum_initial, op=dist.ReduceOp.SUM)
                    dist.all_reduce(intersection_step, op=dist.ReduceOp.SUM)
                    dist.all_reduce(union_step, op=dist.ReduceOp.SUM)
                    dist.all_reduce(intersection_initial, op=dist.ReduceOp.SUM)
                    dist.all_reduce(union_initial, op=dist.ReduceOp.SUM)
                    dist.all_reduce(numel_sum, op=dist.ReduceOp.SUM)

                eps = 1e-12
                self.mask_metrics['step_hamming'] += float(flip_sum_step.float() / (numel_sum.float() + eps))
                self.mask_metrics['initial_hamming'] += float(flip_sum_initial.float() / (numel_sum.float() + eps))
                self.mask_metrics['step_iou'] += float(intersection_step.float() / (union_step.float() + eps))
                self.mask_metrics['initial_iou'] += float(intersection_initial.float() / (union_initial.float() + eps))

            if admm_groups > 0:
                self.mask_metrics['step_hamming'] /= admm_groups
                self.mask_metrics['initial_hamming'] /= admm_groups
                self.mask_metrics['step_iou'] /= admm_groups
                self.mask_metrics['initial_iou'] /= admm_groups

        @torch.no_grad()
        def step(self, closure=None):
            """
            1) (Trainer did backward and clipping)
            2) _proximal_update() adds proximal term to grad
            3) super().step() uses combined grad
            4) _dual_update() for z/u
            """
            self._proximal_update()
            out = super().step(closure)
            self._dual_update()
            self.current_step += 1
            return out

        @torch.no_grad()
        def final_projection(self):
            """
            Apply the final projection to ADMM-tagged parameter groups (in-place).
            This should be called after training is complete to ensure weights have the desired sparsity structure.
            """
            for g in self.param_groups:
                if not g.get("admm", False):
                    continue
                for w in g["params"]:
                    st = self.state[w]
                    importance = None
                    if self.projection_mode == "momentum":
                        v_t = st.get("exp_avg_sq")
                        if self.projection_bias_correction:
                            beta2 = g.get('betas', (0.9, 0.95))[1]
                            importance = v_t / (1.0 - beta2**(st.get("step", 1)))
                        else:
                            importance = v_t
                        if isinstance(importance, DTensor):
                            importance = importance.redistribute(placements=[Replicate()]).to_local()
        
        
                    wnew = self.projection([w.detach()], st["sparsity"], self.prune_n, self.prune_m,
                                           [importance], comparison_group="layer")[0]
                    w.data.copy_(wnew)

        def get_mask_metrics(self) -> Dict[str, float]:
            """
            Return the averaged mask metrics computed at the last interval update.
            """
            return self.mask_metrics

        def get_lmda_stats(self) -> Dict[str, float]:
            """
            Calculates and returns statistics (average, min, max) of per-parameter lmda values.
            """
            total_lmda = 0.0
            count = 0
            min_lmda = float('inf')
            max_lmda = float('-inf')

            for g in self.param_groups:
                if not g.get("admm", False):
                    continue
                for w in g["params"]:
                    if w in self.state:
                        lmda_val = self.state[w].get("lmda")
                        if lmda_val is not None:
                            total_lmda += lmda_val
                            count += 1
                            min_lmda = min(min_lmda, lmda_val)
                            max_lmda = max(max_lmda, lmda_val)
            
            if count == 0:
                return {"avg_lmda": 0.0, "min_lmda": 0.0, "max_lmda": 0.0}
            else:
                return {"avg_lmda": total_lmda / count, "min_lmda": min_lmda, "max_lmda": max_lmda}

    return ADMMOptimizer


class MaskedAdam(torch.optim.Adam):
    """
    A variant of Adam that applies a fixed mask to the parameters after each
    optimizer step. This is useful for retraining pruned models, ensuring that
    the pruned weights remain zero.
    """
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)

    def _lazy_init_mask(self, p,group):
        st = self.state[p]
        if len(st) == 0: ## optimizer state init
            mask = (p.to(torch.float32)!=0.0).bool()
            st['mask'] = mask
            if group["fused"]:
                _device_dtype_check_for_fused(p)
            st["step"] = (
                torch.zeros(
                    (),
                    dtype=_get_scalar_dtype(is_fused=group["fused"]),
                    device=p.device,
                )
                if group["capturable"] or group["fused"]
                else torch.tensor(0.0, dtype=_get_scalar_dtype())
            )
            # Exponential moving average of gradient values
            st["exp_avg"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            # Exponential moving average of squared gradient values
            st["exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            if group["amsgrad"]:
                # Maintains max of all exp. moving avg. of sq. grad. values
                st["max_exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                    )

    @torch.no_grad()
    def step(self, closure=None):
        ## apply mask before step
        for group in self.param_groups:
            for p in group['params']:
                self._lazy_init_mask(p,group)
                if 'mask' in self.state[p]:
                    mask = self.state[p]['mask']
                    p.data.mul_(mask) ## param masking
                    if p.grad is not None:
                        p.grad.data.mul_(mask) ## grad masking
                    if 'exp_avg' in self.state[p]:
                        self.state[p]['exp_avg'].mul_(mask) ## first moment masking
        super().step(closure)
