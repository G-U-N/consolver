# Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
#
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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, is_scipy_available, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from factor_net_ppo import FactorNetPPO

if is_scipy_available():
    import scipy.stats

logger = logging.get_logger(__name__)

@dataclass
class FMPPOSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep.
        actions (`torch.FloatTensor`, *optional*):
            Coefficients predicted by FactorNetPPO for combining model outputs.
        probs (`torch.FloatTensor`, *optional*):
            Probabilities of the sampled actions from FactorNetPPO.
        conds (`Dict`, *optional*):
            Conditioning inputs used by FactorNetPPO.
        masks (`torch.FloatTensor`, *optional*):
            Masks applied to actions to handle variable history lengths.
    """
    prev_sample: torch.FloatTensor
    actions: Optional[torch.FloatTensor] = None
    probs: Optional[torch.FloatTensor] = None
    conds: Optional[Dict] = None
    masks: Optional[torch.FloatTensor] = None

class FMPPOScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler with learnable coefficients provided by a FactorNet for flow-matching.

    This scheduler uses a FactorNet to predict coefficients for combining previous model outputs
    (flow estimates) to predict the next sample, effectively making the integration step learnable.
    The order of the multi-step method is controlled by `order_dim`.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to apply timestep shifting on-the-fly based on the image resolution.
        base_shift (`float`, defaults to 0.5):
            Value to stabilize image generation.
        max_shift (`float`, defaults to 1.15):
            Value change allowed to latent vectors.
        base_image_seq_len (`int`, defaults to 256):
            The base image sequence length.
        max_image_seq_len (`int`, defaults to 4096):
            The maximum image sequence length.
        invert_sigmas (`bool`, defaults to False):
            Whether to invert the sigmas.
        shift_terminal (`float`, defaults to None):
            The end value of the shifted timestep schedule.
        use_karras_sigmas (`bool`, defaults to False):
            Whether to use Karras sigmas for step sizes.
        use_exponential_sigmas (`bool`, defaults to False):
            Whether to use exponential sigmas for step sizes.
        use_beta_sigmas (`bool`, defaults to False):
            Whether to use beta sigmas for step sizes.
        time_shift_type (`str`, defaults to "exponential"):
            The type of dynamic resolution-dependent timestep shifting to apply.
        stochastic_sampling (`bool`, defaults to False):
            Whether to use stochastic sampling.
        order_dim (`int`, defaults to 4):
            The order of the pseudo multi-step method. Defines how many previous steps are used.
        scaler_dim (`int`, defaults to 2):
            Number of scaling factors for the model output and sample.
        use_conv (`bool`, defaults to False):
            Whether to use convolutional layers in FactorNet.
        ppo_type (`str`, defaults to "discrete"):
            Type of PPO network ("discrete" or "continuous").
        factor_net_kwargs (`dict`, *optional*):
            Keyword arguments for the `FactorNetPPO` constructor.
    """
    _compatibles = []
    order = 1  # Dynamically managed by order_dim

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
        order_dim: int = 4,
        scaler_dim: int = 2,
        mu_dim: int = 1,
        use_conv: bool = False,
        ppo_type: str = "discrete",
        factor_net_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                "Only one of `use_beta_sigmas`, `use_exponential_sigmas`, `use_karras_sigmas` can be used."
            )
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")

        # Initialize sigmas and timesteps
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self._shift = shift
        self._step_index = None
        self._begin_index = None

        # Initialize FactorNetPPO
        self.ets = []  # History of model outputs (flow estimates)
        factor_net_kwargs = factor_net_kwargs if factor_net_kwargs is not None else {}
        factor_net_kwargs["order_dim"] = order_dim
        factor_net_kwargs["scaler_dim"] = scaler_dim
        factor_net_kwargs["mu_dim"] = mu_dim
        factor_net_kwargs["use_conv"] = use_conv
        factor_net_kwargs.setdefault("embedding_dim", 32)
        factor_net_kwargs.setdefault("hidden_dim", 256)

        if ppo_type == "discrete":
            factor_net_kwargs.setdefault("num_actions", 161)
            self.factor_net = FactorNetPPO(**factor_net_kwargs)
        else:
            assert 0
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided"
                )
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps
        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)

        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None
        self.ets = []  # Reset history for new run
        self._curr_sigma = None
        
        

    def set_default_coefficients(self, action_params, scale_params, num_ets):
        """
        Normalizes action parameters to sum to 1 and adjusts scale parameters.

        Args:
            action_params (`List[torch.FloatTensor]`): Coefficients for combining model outputs.
            scale_params (`List[torch.FloatTensor]`): Scaling factors.
            num_ets (`int`): Number of available model outputs in history.

        Returns:
            Tuple[List[torch.FloatTensor], List[torch.FloatTensor]]: Normalized action and scale parameters.
        """
        action_params.append(action_params[-1])  # Placeholder
        action_params[0] = action_params[0] + 1

        if num_ets > 1:
            action_params[num_ets - 1] = 1 - torch.sum(torch.stack(action_params[:num_ets - 1]), dim=0)

        scale_params = [scale + 1 for scale in scale_params]
        return action_params, scale_params

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index
    
    @property
    def shift(self):
        """
        The value used for shifting.
        """
        return self._shift
    
    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index
    
    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_shift(self, shift: float):
        self._shift = shift

        
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[FMPPOSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep using a learnable multi-step method.

        Args:
            model_output (`torch.FloatTensor`): Direct output from the flow-matching model.
            timestep (`float` or `torch.FloatTensor`): Current timestep in the diffusion chain.
            sample (`torch.FloatTensor`): Current sample (x_t).
            s_churn (`float`): Churn parameter for stochastic sampling.
            s_tmin (`float`): Minimum timestep for churn.
            s_tmax (`float`): Maximum timestep for churn.
            s_noise (`float`): Scaling factor for noise in stochastic sampling.
            generator (`torch.Generator`, *optional*): Random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*): Per-token timesteps.
            return_dict (`bool`): Whether to return a dataclass or tuple.

        Returns:
            [`FMPPOSchedulerOutput`] or `tuple`:
                Contains the previous sample (`prev_sample`) and optionally actions, probs, conds, and masks.
        """
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is 'None'. Call 'set_timesteps' first.")

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer indices as timesteps to `step()` is not supported. Pass one of `scheduler.timesteps`."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues
        sample = sample.to(torch.float32)

        # Store current model output (flow estimate)
        current_et = model_output
        self.ets.append(current_et)
        self.ets = self.ets[-self.config.order_dim:]  # Limit history
        num_ets = len(self.ets)

        # # Calculate current and next sigma
        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps
            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)
            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self.step_index
            current_sigma = self.sigmas[sigma_idx]
            next_sigma = self.sigmas[sigma_idx + 1]
            dt = next_sigma - current_sigma
        
        if self._curr_sigma is None:
            self._curr_sigma = self.sigmas[0].to(model_output.device).repeat(model_output.shape[0])

        # Get learnable coefficients from FactorNet
        batch_size = model_output.shape[0]
        conds = torch.tensor([[current_sigma, next_sigma]], dtype=model_output.dtype, device=model_output.device)
        # next_sigma 
        conds = conds.repeat(batch_size, 1) # not need to repeat

        ets_tensor = torch.stack(self.ets[::-1], dim=1)  # [B, num_ets, ...]
        if num_ets < self.config.order_dim:
            padding = torch.zeros(
                batch_size,
                self.config.order_dim - num_ets,
                *model_output.shape[1:],
                dtype=model_output.dtype,
                device=model_output.device
            )
            ets_tensor = torch.cat([ets_tensor, padding], dim=1)

        conds_dict = {'x': conds, 'epsilon': ets_tensor}
        factor_net = self.factor_net.module if hasattr(self.factor_net, "module") else self.factor_net
        actions, probs = factor_net.sample_action(conds_dict)
        

        masks = torch.ones_like(probs)
        masks[:, num_ets - 1:self.config.order_dim - 1] = 0

        action_params = [actions[:, i].unsqueeze(1).unsqueeze(2) if i < self.config.order_dim + self.config.scaler_dim - 1 else actions[:, i]
                         for i in range(self.config.order_dim + self.config.scaler_dim + self.config.mu_dim - 1)]

        action_params, scale_params, mu_params = action_params[:self.config.order_dim - 1], action_params[self.config.order_dim - 1:self.config.order_dim + self.config.scaler_dim - 1], action_params[self.config.order_dim + self.config.scaler_dim - 1:]
        action_params, scale_params = self.set_default_coefficients(action_params, scale_params, num_ets)

        # Combine model outputs using learned coefficients
        if num_ets == 1:
            effective_model_output = self.ets[-1]
        else:
            coeffs_to_use = action_params[:num_ets]
            ets_to_use = self.ets[::-1]
            effective_model_output = sum(c * e for c, e in zip(coeffs_to_use, ets_to_use))

        # Apply scaling factors
        if len(scale_params) == 1:
            effective_model_output = effective_model_output * scale_params[0]
        elif len(scale_params) == 2:
            effective_model_output = effective_model_output * scale_params[0]
            sample = sample * scale_params[1]
        elif len(scale_params) > 0:
            raise NotImplementedError("More than two scale parameters not supported.")

        prev_sample = sample + dt * effective_model_output

        # Increment step index
        self._step_index += 1

        # Cast back to model-compatible dtype
        if per_token_timesteps is None:
            prev_sample = prev_sample.to(model_output.dtype)

        # Debug info
        if num_ets > 0:
            action_vals = [f"{p[0].item():.3f}" for p in action_params[:num_ets]] + [f"{p[0].item():.3f}" for p in scale_params] + [f"{p[0].item():.3f}" for p in mu_params]
            print(f"T={self._sigma_to_t(current_sigma.item()):.2f} -> {self._sigma_to_t(next_sigma.item()):.2f} | "
                  f"Coeffs: [{' '.join(action_vals)}] | Prob: {probs[0]}")

        self._curr_sigma = next_sigma

        if not return_dict:
            return (prev_sample, actions, probs, conds_dict, masks)

        return FMPPOSchedulerOutput(
            prev_sample=prev_sample,
            actions=actions,
            probs=probs,
            conds=conds_dict,
            masks=masks
        )

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Forward process in flow-matching (unchanged)."""
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)
        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample
        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        sigma_min = self.config.sigma_min if hasattr(self.config, "sigma_min") else in_sigmas[-1].item()
        sigma_max = self.config.sigma_max if hasattr(self.config, "sigma_max") else in_sigmas[0].item()
        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _convert_to_exponential(self, in_sigmas: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        sigma_min = self.config.sigma_min if hasattr(self.config, "sigma_min") else in_sigmas[-1].item()
        sigma_max = self.config.sigma_max if hasattr(self.config, "sigma_max") else in_sigmas[0].item()
        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    def _convert_to_beta(self, in_sigmas: torch.Tensor, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6) -> torch.Tensor:
        sigma_min = self.config.sigma_min if hasattr(self.config, "sigma_min") else in_sigmas[-1].item()
        sigma_max = self.config.sigma_max if hasattr(self.config, "sigma_max") else in_sigmas[0].item()
        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return sigmas

    def _time_shift_exponential(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu, sigma, t):
        return mu / (mu + (1 / t - 1) ** sigma)

    def __len__(self):
        return self.config.num_train_timesteps



