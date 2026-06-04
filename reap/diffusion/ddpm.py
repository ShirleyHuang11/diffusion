"""Gaussian diffusion (DDPM) over trajectory windows with inpainting sampling.

Conditioning is by inpainting: pinned window positions are overwritten with
their (appropriately noised) known values at every denoising step, so the
final sample matches the pins exactly. Classifier-free guidance mixes the
conditional and null-condition noise predictions when a condition vector and
a guidance scale are provided.
"""

from __future__ import annotations

import numpy as np
import torch

from reap.diffusion.model import TrajectoryDenoiser


class GaussianDiffusion:
    def __init__(self, num_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def training_loss(
        self,
        model: TrajectoryDenoiser,
        x0: torch.Tensor,
        cond: torch.Tensor | None = None,
        cond_dropout: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.num_steps, (b,), generator=generator)
        noise = torch.randn(x0.shape, generator=generator)
        noisy = self.q_sample(x0, t, noise)
        use_cond = cond
        if cond is not None and cond_dropout > 0:
            if torch.rand((), generator=generator).item() < cond_dropout:
                use_cond = None  # train the null-condition path for CFG
        predicted = model(noisy, t, use_cond)
        return ((predicted - noise) ** 2).mean()

    def _predict_noise(
        self,
        model: TrajectoryDenoiser,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None,
        guidance_scale: float,
    ) -> torch.Tensor:
        if cond is None or guidance_scale == 1.0 or model.cond_dim == 0:
            return model(x, t, cond if model.cond_dim > 0 else None)
        eps_null = model(x, t, None)
        eps_cond = model(x, t, cond)
        return eps_null + guidance_scale * (eps_cond - eps_null)

    @torch.no_grad()
    def sample(
        self,
        model: TrajectoryDenoiser,
        n: int,
        pin: dict[int, torch.Tensor] | None = None,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw ``n`` windows of shape (n, W, D) in normalized space.

        ``pin`` maps window indices to (D,) or (n, D) tensors of known
        normalized states (e.g. {0: s_t} forward, {0: s_t, W-1: g} bridges).
        """
        w, d = model.window, model.state_dim
        pin = {
            int(i): (v if v.dim() == 2 else v.unsqueeze(0).expand(n, -1)).float()
            for i, v in (pin or {}).items()
        }
        for index in pin:
            if not 0 <= index < w:
                raise ValueError(f"pin index {index} outside window [0, {w})")

        x = torch.randn((n, w, d), generator=generator)
        for step in reversed(range(self.num_steps)):
            t = torch.full((n,), step, dtype=torch.long)
            # keep pinned positions on the known-trajectory manifold at this noise level
            for index, value in pin.items():
                noise = torch.randn((n, d), generator=generator)
                x[:, index] = self.q_sample(
                    value.unsqueeze(1), t, noise.unsqueeze(1)
                ).squeeze(1)
            eps = self._predict_noise(model, x, t, cond, guidance_scale)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bar[step]
            mean = (x - (1 - alpha) / (1 - alpha_bar).sqrt() * eps) / alpha.sqrt()
            if step > 0:
                x = mean + self.betas[step].sqrt() * torch.randn(
                    (n, w, d), generator=generator
                )
            else:
                x = mean
        for index, value in pin.items():
            x[:, index] = value  # exact pins on the final sample
        return x


def train_teacher(
    model: TrajectoryDenoiser,
    diffusion: GaussianDiffusion,
    dataset,
    steps: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    cond_fn=None,
    seed: int = 0,
) -> list[float]:
    """Simple training loop; returns the per-step loss history.

    ``cond_fn(batch) -> (B, cond_dim) tensor`` supplies condition vectors
    (e.g. policy embeddings) when the model is conditional.
    """
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        batch = dataset.sample_batch(batch_size, rng)
        cond = cond_fn(batch) if cond_fn is not None else None
        loss = diffusion.training_loss(model, batch, cond, generator=generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return history
