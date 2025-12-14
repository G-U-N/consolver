# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

from factor_net_ppo import FactorNetPPO
from factor_net_ppo_continous import FactorNetPPOContinous

def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """Creates a beta schedule based on the specified alpha_t_bar function."""
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class PPOScheduler(SchedulerMixin, ConfigMixin):
    """
    A pseudo-linear multi-step scheduler with learnable coefficients provided by a FactorNet.

    This scheduler uses a FactorNet to predict coefficients for combining previous model outputs
    (noise estimates) to predict the next sample, effectively making the integration step learnable.
    The order of the multi-step method is controlled by `order_dim`.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`].

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            Number of diffusion steps used during training.
        beta_start (`float`, defaults to 0.0001): Initial beta value.
        beta_end (`float`, defaults to 0.02): Final beta value.
        beta_schedule (`str`, defaults to `"linear"`):
            Beta schedule type (`linear`, `scaled_linear`, `squaredcos_cap_v2`).
        trained_betas (`np.ndarray`, *optional*): Pre-computed betas.
        prediction_type (`str`, defaults to `epsilon`):
            Type of prediction (`epsilon` or `v_prediction`).
        timestep_spacing (`str`, defaults to `"leading"`): Timestep spacing strategy.
        steps_offset (`int`, defaults to 0): Offset added to inference steps.
        order_dim (`int`, defaults to 4):
            The order of the pseudo multi-step method. Defines how many previous steps
            are used. `action_dims` for the FactorNet will be `order_dim + 1`.
        factor_net_kwargs (`dict`, *optional*):
            Keyword arguments for the `FactorNetPPO` constructor. `action_dims` will
            be set to `order_dim + 1` if not provided, or validated if it is.
    """
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # order is dynamically managed by order_dim, setting a placeholder
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        order_dim: int = 4,
        scaler_dim: int = 2,
        use_conv = False,
        ppo_type = "discrete",
        factor_net_kwargs: Optional[Dict] = None,
    ):
        # --- Beta and Alpha Setup ---
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} schedule not implemented.")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Required for final step: alpha_t=0 -> alpha_prod_t=0
        # Previously set_alpha_to_one=False
        self.final_alpha_cumprod = self.alphas_cumprod[0]

        # --- Standard Diffuser Scheduler Setup ---
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.timesteps = torch.from_numpy(self._timesteps) # Initialize with default

        # --- Learnable Scheduler Specific Setup ---
        self.ets = [] # History of model outputs (noise estimates)

        # Initialize FactorNetPPO
        factor_net_kwargs = factor_net_kwargs if factor_net_kwargs is not None else {}

        factor_net_kwargs["order_dim"] = order_dim
        factor_net_kwargs["scaler_dim"] = scaler_dim
        factor_net_kwargs["use_conv"] = use_conv
        # Default kwargs for FactorNet if not provided (as per user prompt)
        factor_net_kwargs.setdefault("embedding_dim", 32)
        factor_net_kwargs.setdefault("hidden_dim", 256)

        if ppo_type == "discrete":
            factor_net_kwargs.setdefault("num_actions", 161) # Assuming this is discretization bins
            self.factor_net = FactorNetPPO(**factor_net_kwargs)
        else:
            self.factor_net = FactorNetPPOContinous(**factor_net_kwargs)


    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """Sets the discrete timesteps used for the diffusion chain."""
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(f"`num_inference_steps` ({num_inference_steps}) cannot be larger than `num_train_timesteps` ({self.config.num_train_timesteps}).")

        self.num_inference_steps = num_inference_steps

        # Recalculate timesteps based on spacing strategy
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype(np.int64)
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64) - 1
        else:
            raise ValueError(f"Unsupported timestep_spacing: {self.config.timestep_spacing}.")

        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.ets = [] # Reset history for new run

    def set_default_coefficients(self, action_params, scale_params, num_ets):
        action_params.append(action_params[-1]) # just a place holder
        

        action_params[0] = action_params[0] + 1

        if num_ets > 1:
            action_params[num_ets - 1] = 1 - torch.sum(torch.stack(action_params[:num_ets - 1]), dim=0) # summation equals 1 

        scale_params = [scale + 1 for scale in scale_params]
        return action_params, scale_params


    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predicts the sample at the previous timestep using a learnable PLMS-like method.

        Args:
            model_output (`torch.FloatTensor`): Direct output from the diffusion model (noise estimate).
            timestep (`int`): Current timestep in the diffusion chain.
            sample (`torch.FloatTensor`): Current sample (x_t).
            return_dict (`bool`, defaults to `True`): Return type choice.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                Contains the previous sample (`prev_sample`) and potentially other info like
                the actions sampled from `FactorNet`.
        """
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is 'None'. Call 'set_timesteps' first.")

        # --- Calculate Previous Timestep ---
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # --- Get Learnable Coefficients from FactorNet ---
        batch_size = model_output.shape[0]
        conds = torch.tensor([[timestep, prev_timestep]], dtype=model_output.dtype, device=model_output.device)

        # conds = torch.tensor([[timestep, prev_timestep]], dtype=model_output.dtype, device=model_output.device)
        conds = conds.repeat(batch_size, 1)

        # --- Apply Learnable Pseudo-Linear Multi-Step ---
        # Store current model output (noise estimate)
        current_et = model_output
        self.ets.append(current_et)

        # Limit history size based on order_dim
        self.ets = self.ets[-self.config.order_dim:]
        num_ets = len(self.ets)


        ets_tensor = torch.stack(self.ets[::-1], dim=1)  # [B, num_ets, ...]
        # Pad with zeros if less than order_dim
        if num_ets < self.config.order_dim:
            padding = torch.zeros(
                batch_size,
                self.config.order_dim - num_ets,
                *model_output.shape[1:],  # Match spatial dimensions
                dtype=model_output.dtype,
                device=model_output.device
            )
            ets_tensor = torch.cat([ets_tensor, padding], dim=1)  # [B, order_dim, ...]

        conds = {
            'x': conds,  # [B, 2]
            'epsilon': ets_tensor  # [B, order_dim, C, H, W] or equivalent
        }

        factor_net = self.factor_net.module if hasattr(self.factor_net, "module") else self.factor_net
        actions, probs = factor_net.sample_action(conds)
        
        # actions[:, 0] = 0.4
        print(actions)
        
        # assert 0


        masks = torch.ones_like(probs)
        masks[:, num_ets - 1:self.config.order_dim-1] = 0 # only the actions of previous num_ets - 1 are used. 
         

        # These are p0, p1, ..., p_{order_dim-1}, scale_factor_1, scale_factor_2
        action_params = [actions[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                         for i in range(self.config.order_dim + self.config.scaler_dim -1)]
        action_params, scale_params = action_params[:self.config.order_dim-1], action_params[self.config.order_dim-1:]



        action_params, scale_params = self.set_default_coefficients(action_params, scale_params, num_ets)


        # Apply the learnable linear combination of past estimates
        if num_ets == 1:
            # First step: Use current estimate directly
            effective_model_output = self.ets[-1]
        else:
            # Combine estimates using learned coefficients
            # Only use available history and corresponding coefficients
            coeffs_to_use = action_params[:num_ets]
            ets_to_use = self.ets[::-1] # Reverse for alignment (ets[-1] with param[0], etc.)

            effective_model_output = sum(c * e for c, e in zip(coeffs_to_use, ets_to_use))

        if len(scale_params) == 1:
            effective_model_output = effective_model_output * scale_params[0]
        elif len(scale_params) == 2:
            effective_model_output = effective_model_output * scale_params[0]
            sample = sample * scale_params[1]
        elif len(scale_params) > 0:
            assert 0, "not implemented"


        prev_sample = self._get_prev_sample(sample, timestep, prev_timestep, effective_model_output)


        # --- Optional: Print Debug Info ---
        if num_ets > 0: # Only print if we have coefficients
             action_vals = [f"{p[0].item():.3f}" for p in action_params[:num_ets]] + [f"{p[0].item():.3f}" for p in scale_params]
             print(f"T={timestep} -> {prev_timestep} | "
                   f"Coeffs: [{' '.join(action_vals)}] | "
                   f"Prob: {probs[0]}")


        # --- Return Results ---
        if not return_dict:
            # Note: Returning actions, probs, conds for potential PPO training loop
            return (prev_sample, actions, probs, conds, masks)

        return SchedulerOutput(prev_sample=prev_sample, actions=actions, probs=probs, conds=conds, masks=masks)


    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """No scaling needed for this scheduler type."""
        return sample

    def _get_prev_sample(self, sample, timestep, prev_timestep, model_output):
        """Calculates x_{t-1} from x_t and model_output using the DDIM formula."""
        # Get alpha/beta products for current and previous timesteps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev


        # Compute variance prediction (if needed)
        if self.config.prediction_type == "v_prediction":
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        elif self.config.prediction_type != "epsilon":
            raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")


        # 1. Calculate estimated original sample (x_0_hat)
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 2. Calculate coefficients for clipping (if needed, not implemented here)
        # pred_original_sample = torch.clamp(pred_original_sample, -1, 1) # Optional clipping

        # 3. Calculate x_{t-1} using x_0_hat
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + \
                      (beta_prod_t_prev) ** (0.5) * model_output 

        return prev_sample


    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """Adds noise to samples according to the noise schedule."""
        # Ensure device and dtype consistency
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps