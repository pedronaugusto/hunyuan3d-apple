"""
MLX flow-matching scheduler and sampler helpers for Hunyuan3D.

This mirrors the upstream `hy3dshape.schedulers.FlowMatchEulerDiscreteScheduler`
API so the MLX shape pipeline can follow the same scheduler-driven call flow.
"""
import mlx.core as mx
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace


class FlowMatchEulerDiscreteSchedulerOutput:
    def __init__(self, prev_sample):
        self.prev_sample = prev_sample


class MlxFlowMatchEulerDiscreteScheduler:
    """MLX-native mirror of the upstream FlowMatchEulerDiscreteScheduler."""

    order = 1

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.config = SimpleNamespace(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=False,
        )
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32).copy()
        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = mx.array(sigmas * num_train_timesteps, dtype=mx.float32)
        self.sigmas = mx.array(sigmas, dtype=mx.float32)
        self.sigma_min = float(sigmas[-1])
        self.sigma_max = float(sigmas[0])
        self._step_index = None
        self._begin_index = None
        self.init_noise_sigma = 1.0

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int = None, device=None, sigmas=None, mu=None):
        del device, mu
        if sigmas is None:
            timesteps = np.linspace(
                self.sigma_max * self.config.num_train_timesteps,
                self.sigma_min * self.config.num_train_timesteps,
                num_inference_steps,
                dtype=np.float32,
            )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.asarray(sigmas, dtype=np.float32)

        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
        self.timesteps = mx.array(sigmas * self.config.num_train_timesteps, dtype=mx.float32)
        self.sigmas = mx.array(np.concatenate([sigmas, [1.0]], axis=0), dtype=mx.float32)
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        ts = np.array(schedule_timesteps)
        t = float(np.array(timestep).reshape(-1)[0])
        indices = np.where(np.isclose(ts, t))[0]
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos])

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(self, model_output, timestep, sample, return_dict: bool = True, **kwargs):
        del kwargs
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        prev_sample = sample.astype(mx.float32) + (sigma_next - sigma) * model_output.astype(mx.float32)
        prev_sample = prev_sample.astype(model_output.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


class FlowEulerSampler:
    """Euler ODE sampler for Hunyuan3D flow matching with classifier-free guidance."""

    def _inference_model(self, model, x_t, sigma, cond, **kwargs):
        """Run model forward pass. sigma ∈ [0, 1] passed directly as timestep."""
        batch_size = x_t.shape[0]
        t = mx.array([sigma] * batch_size, dtype=mx.float32)
        return model(x_t, t, cond, **kwargs)

    def sample(self, model, noise, cond, neg_cond,
               steps: int = 50, guidance_strength: float = 5.0,
               shift: float = 1.0,
               verbose: bool = True, **kwargs):
        """
        Generate samples using Euler method with CFG.

        Matches upstream FlowMatchingPipeline exactly:
        - sigmas = linspace(0, 1, steps) with optional shift
        - Sentinel sigma=1.0 appended (last step: sigma=1→1, dt=0, no-op)
        - Step: sample = sample + (sigma_next - sigma) * velocity
        - CFG: uncond + guidance * (cond - uncond)
        """
        sample = noise

        # Upstream: sigmas = np.linspace(0, 1, num_inference_steps)
        # Then passed to scheduler.set_timesteps(sigmas=sigmas)
        # With shift=1.0 (default), shift formula is identity
        sigmas = np.linspace(0.0, 1.0, steps)

        # Apply shift: sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        if shift != 1.0:
            sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

        # Sentinel (dt=0 on final step)
        sigmas = np.append(sigmas, 1.0).tolist()

        for i in tqdm(range(steps), desc="Diffusion Sampling", disable=not verbose):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            if guidance_strength > 0:
                # Match upstream exactly: duplicate latents and run conditional and
                # unconditional branches in a single batched forward.
                model_input = mx.concatenate([sample, sample], axis=0)
                cond_input = {
                    key: mx.concatenate([cond[key], neg_cond[key]], axis=0)
                    for key in cond.keys()
                }
                pred = self._inference_model(model, model_input, sigma, cond_input, **kwargs)
                pred_cond, pred_uncond = mx.split(pred, 2, axis=0)
                pred_v = pred_uncond + guidance_strength * (pred_cond - pred_uncond)
            else:
                pred_v = self._inference_model(model, sample, sigma, cond, **kwargs)

            dt = sigma_next - sigma
            sample = sample + dt * pred_v

            mx.eval(sample)

            if i < 3 or i == steps - 1:
                print(f"  step {i}: sigma={sigma:.4f}→{sigma_next:.4f}, "
                      f"v_mean={pred_v.mean().item():.4f}, v_std={pred_v.var().item()**.5:.4f}, "
                      f"x_mean={sample.mean().item():.4f}, x_std={sample.var().item()**.5:.4f}")

        return sample
