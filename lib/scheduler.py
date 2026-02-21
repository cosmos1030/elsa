import math
from absl import logging


class PenaltyScheduler:

    def __init__(self, optimizer,initial_lmda, final_lmda, total_steps, mode='linear'):
        """
        A scheduler for penalty parameter (lmda).
        
        Args:
            optimizer (torch.optim.Optimizer) : optimizer for which the scheduler is used
            initial_lmda (float): initial value of lambda
            final_lmda (float): final value of lambda
            total_steps (int): Total number of steps for scheduling.
            mode (str): Scheduling mode ( 'constant','linear','cosine').
        """
        self.optimizer = optimizer
        self.total_steps = total_steps

        self.mode = mode.lower()
        self.initial_lmda = initial_lmda
        self.final_lmda = final_lmda
        self.step_count = 0

    def step(self):
        """Update the scheduler step."""
        self.step_count += 1
        self.step_count = min(self.step_count, self.total_steps)  # Clamp to total_steps

        new_lmda = self._calculate_lmda_for_fixed_modes()
        for group in self.optimizer.param_groups:
            if group.get('admm', False):
                group['lmda'] = new_lmda
            
    def _calculate_lmda_for_fixed_modes(self):
        """Calculate the current lmda based on the mode for non-adaptive modes."""
        init_lmda = self.initial_lmda
        final_lmda = self.final_lmda

        if self.mode == 'linear':
            return init_lmda + (final_lmda - init_lmda) * (self.step_count / self.total_steps)
        
        elif self.mode == 'constant':
            return final_lmda
        
        elif self.mode == 'cosine':
            return init_lmda + (final_lmda - init_lmda) * 0.5 * (1 - math.cos(math.pi * self.step_count / self.total_steps))
        
        elif self.mode == 'log':
            log_step = math.log(1 + self.step_count)
            log_total = math.log(1 + self.total_steps)
            return init_lmda + (final_lmda - init_lmda) * (log_step / log_total)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
