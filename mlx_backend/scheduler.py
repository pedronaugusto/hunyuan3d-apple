"""
UniPC Multistep Scheduler for MLX diffusion inference.

Ported from diffusers UniPCMultistepScheduler — 2nd-order multistep ODE solver
with predictor-corrector (UniP + UniC). Matches upstream Hunyuan3D texture pipeline.

Config (from scheduler_config.json):
  prediction_type: v_prediction
  beta_schedule: scaled_linear [0.00085, 0.012]
  timestep_spacing: trailing
  rescale_betas_zero_snr: true
  num_train_timesteps: 1000
"""
import math
import numpy as np


_LOG_EPS = 1e-12


def _rescale_zero_terminal_snr(betas: np.ndarray) -> np.ndarray:
    """Rescale betas to have zero terminal SNR (Algorithm 1, arXiv:2305.08891)."""
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_bar_sqrt = np.sqrt(alphas_cumprod)

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0]
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1]

    alphas_bar_sqrt = alphas_bar_sqrt - alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    alphas_bar = alphas_bar_sqrt ** 2
    alphas = np.concatenate([alphas_bar[0:1], alphas_bar[1:] / alphas_bar[:-1]])
    betas = 1.0 - alphas
    return betas


class UniPCMultistepScheduler:
    """UniPC multistep scheduler ported to MLX/numpy.

    2nd-order predictor-corrector using B(h) formulation (bh2 solver type).
    Operates on numpy for schedule math; actual latent tensors use mx.array
    externally (the step() method works with any array-like via numpy).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "v_prediction",
        solver_order: int = 2,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: list = None,
        rescale_betas_zero_snr: bool = True,
    ):
        self.prediction_type = prediction_type
        self.solver_order = solver_order
        self.predict_x0 = predict_x0
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.disable_corrector = disable_corrector or []
        self.num_train_timesteps = num_train_timesteps

        # Beta schedule
        if beta_schedule == "scaled_linear":
            betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        if rescale_betas_zero_snr:
            betas = _rescale_zero_terminal_snr(betas)

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas).astype(np.float64)

        if rescale_betas_zero_snr:
            # Close to 0 without being 0 so first sigma is not inf
            self.alphas_cumprod[-1] = 2 ** -24

        self.alpha_t = np.sqrt(self.alphas_cumprod)
        self.sigma_t = np.sqrt(1.0 - self.alphas_cumprod)
        self.lambda_t = np.log(self.alpha_t) - np.log(self.sigma_t)
        self.sigmas = ((1.0 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # Standard deviation of initial noise distribution
        self.init_noise_sigma = 1.0

        # State (set by set_timesteps)
        self.timesteps = None
        self.num_inference_steps = None
        self._sigmas = None  # sigmas array indexed by step
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self.this_order = None

    @property
    def step_index(self):
        return self._step_index

    def set_timesteps(self, num_inference_steps: int):
        """Compute timestep schedule with trailing spacing (matching upstream)."""
        self.num_inference_steps = num_inference_steps

        # Trailing spacing
        step_ratio = self.num_train_timesteps / num_inference_steps
        timesteps = np.arange(self.num_train_timesteps, 0, -step_ratio).round().astype(np.int64)
        timesteps = timesteps - 1

        # Compute sigmas at these timesteps
        all_sigmas = ((1.0 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        sigmas = np.interp(timesteps.astype(np.float64), np.arange(len(all_sigmas)), all_sigmas)
        # Append final sigma = 0
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float64)

        self._sigmas = sigmas
        self.timesteps = timesteps.tolist()

        # Reset state
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self.this_order = None

    def _sigma_to_alpha_sigma_t(self, sigma):
        """Convert sigma to (alpha_t, sigma_t) for VP schedule."""
        alpha_t = 1.0 / (sigma ** 2 + 1.0) ** 0.5
        sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def _safe_lambda(self, sigma):
        """Match UniPC math while avoiding log(0) at the terminal sigma=0 step."""
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        return math.log(max(alpha_t, _LOG_EPS)) - math.log(max(sigma_t, _LOG_EPS))

    def _init_step_index(self, timestep):
        """Initialize step index from timestep."""
        for i, t in enumerate(self.timesteps):
            if t == timestep:
                self._step_index = i
                return
        # Fallback: last step
        self._step_index = len(self.timesteps) - 1

    def convert_model_output(self, model_output, sample):
        """Convert model output to x0 prediction (or epsilon, depending on predict_x0)."""
        import mlx.core as mx

        sigma = self._sigmas[self._step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            elif self.prediction_type == "sample":
                x0_pred = model_output
            else:
                raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
            return x0_pred
        else:
            if self.prediction_type == "epsilon":
                return model_output
            elif self.prediction_type == "v_prediction":
                return alpha_t * model_output + sigma_t * sample
            elif self.prediction_type == "sample":
                return (sample - alpha_t * model_output) / sigma_t
            else:
                raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def multistep_uni_p_bh_update(self, model_output, sample, order):
        """UniP predictor step using B(h) formulation."""
        import mlx.core as mx

        m0 = self.model_outputs[-1]
        x = sample

        sigma_t = self._sigmas[self._step_index + 1]
        sigma_s0 = self._sigmas[self._step_index]
        alpha_t, sigma_t_val = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0_val = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = self._safe_lambda(sigma_t)
        lambda_s0 = self._safe_lambda(sigma_s0)
        h = lambda_t - lambda_s0

        # Build difference approximations from history
        rks = []
        D1s = []
        for i in range(1, order):
            si = self._step_index - i
            mi = self.model_outputs[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self._sigmas[si])
            lambda_si = self._safe_lambda(self._sigmas[si])
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)

        hh = -h if self.predict_x0 else h
        h_phi_1 = math.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1.0

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = math.expm1(hh)
        else:
            raise ValueError(f"Unknown solver_type: {self.solver_type}")

        # Build R matrix and b vector for solving coefficients
        R = []
        b = []
        for i in range(1, order + 1):
            R.append([rk ** (i - 1) for rk in rks])
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i

        if len(D1s) > 0:
            if order == 2:
                rhos_p = [0.5]
            else:
                R_np = np.array([row[:-1] for row in R[:-1]], dtype=np.float64)
                b_np = np.array(b[:-1], dtype=np.float64)
                rhos_p = np.linalg.solve(R_np, b_np).tolist()

            # Compute weighted sum of D1s
            pred_res = rhos_p[0] * D1s[0]
            for k in range(1, len(D1s)):
                pred_res = pred_res + rhos_p[k] * D1s[k]
        else:
            pred_res = None

        if self.predict_x0:
            x_t_ = (sigma_t_val / sigma_s0_val) * x - alpha_t * h_phi_1 * m0
            if pred_res is not None:
                x_t = x_t_ - alpha_t * B_h * pred_res
            else:
                x_t = x_t_
        else:
            x_t_ = (alpha_t / alpha_s0) * x - sigma_t_val * h_phi_1 * m0
            if pred_res is not None:
                x_t = x_t_ - sigma_t_val * B_h * pred_res
            else:
                x_t = x_t_

        return x_t

    def multistep_uni_c_bh_update(self, this_model_output, last_sample, this_sample, order):
        """UniC corrector step using B(h) formulation."""
        import mlx.core as mx

        m0 = self.model_outputs[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t = self._sigmas[self._step_index]
        sigma_s0 = self._sigmas[self._step_index - 1]
        alpha_t, sigma_t_val = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0_val = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = self._safe_lambda(sigma_t)
        lambda_s0 = self._safe_lambda(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = self._step_index - (i + 1)
            mi = self.model_outputs[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self._sigmas[si])
            lambda_si = self._safe_lambda(self._sigmas[si])
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)

        hh = -h if self.predict_x0 else h
        h_phi_1 = math.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1.0

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = math.expm1(hh)
        else:
            raise ValueError(f"Unknown solver_type: {self.solver_type}")

        R = []
        b = []
        for i in range(1, order + 1):
            R.append([rk ** (i - 1) for rk in rks])
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i

        # Solve for corrector coefficients
        if order == 1:
            rhos_c = [0.5]
        else:
            R_np = np.array(R, dtype=np.float64)
            b_np = np.array(b, dtype=np.float64)
            rhos_c = np.linalg.solve(R_np, b_np).tolist()

        D1_t = model_t - m0

        if self.predict_x0:
            x_t_ = (sigma_t_val / sigma_s0_val) * x - alpha_t * h_phi_1 * m0
            if len(D1s) > 0:
                corr_res = sum(rhos_c[k] * D1s[k] for k in range(len(D1s)))
                x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
            else:
                x_t = x_t_ - alpha_t * B_h * (rhos_c[-1] * D1_t)
        else:
            x_t_ = (alpha_t / alpha_s0) * x - sigma_t_val * h_phi_1 * m0
            if len(D1s) > 0:
                corr_res = sum(rhos_c[k] * D1s[k] for k in range(len(D1s)))
                x_t = x_t_ - sigma_t_val * B_h * (corr_res + rhos_c[-1] * D1_t)
            else:
                x_t = x_t_ - sigma_t_val * B_h * (rhos_c[-1] * D1_t)

        return x_t

    def step(self, model_output, timestep, sample):
        """Run one UniPC step: corrector (if applicable) then predictor.

        Args:
            model_output: raw model prediction (before convert_model_output)
            timestep: current timestep (int)
            sample: current noisy sample

        Returns:
            prev_sample: denoised sample at t-1
        """
        if self._step_index is None:
            self._init_step_index(timestep)

        # Corrector step (UniC) — uses current model output to refine previous prediction
        use_corrector = (
            self._step_index > 0
            and (self._step_index - 1) not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)

        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        # Shift model output history
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        # Determine order for this step
        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self._step_index)
        else:
            this_order = self.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)

        # Predictor step (UniP)
        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        self._step_index += 1

        return prev_sample

    def scale_model_input(self, sample, timestep=None):
        """Identity scaling (UniPC doesn't scale inputs)."""
        return sample
