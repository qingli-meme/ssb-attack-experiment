from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class TriggerConfig:
    reference_size: int = 30
    radial_frequencies: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    angular_orders: Tuple[int, ...] = (0, 1, 2, 3)
    contrast: float = 1.75
    radius_floor: float = 0.06
    channels: int = 3
    init_std: float = 0.15


class ScaleSteerableTrigger(nn.Module):
    """Scale-steerable trigger parameterized by shared harmonic coefficients.

    Continuous basis (ignoring the tiny radius regularization region):
        psi_{k,n}^c(r, theta) = cos(omega_k log r + n theta)
        psi_{k,n}^s(r, theta) = sin(omega_k log r + n theta)

    For scale ratio s, the shared coefficient pair [a, b] is analytically
    steered by a 2D rotation R(omega_k log s). Therefore all rendered sizes
    are realizations of the same coefficient vector rather than independent
    trigger templates.

    The module learns only ``coeff``. The basis and the steering rule are fixed.
    """

    def __init__(
        self,
        reference_size: int = 30,
        radial_frequencies: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
        angular_orders: Sequence[int] = (0, 1, 2, 3),
        contrast: float = 1.75,
        radius_floor: float = 0.06,
        channels: int = 3,
        init_std: float = 0.15,
    ) -> None:
        super().__init__()
        if reference_size <= 0:
            raise ValueError("reference_size must be positive")
        if radius_floor <= 0:
            raise ValueError("radius_floor must be > 0 because log(r) is used")
        if channels != 3:
            raise ValueError("This implementation expects RGB triggers (channels=3).")

        self.config = TriggerConfig(
            reference_size=int(reference_size),
            radial_frequencies=tuple(float(x) for x in radial_frequencies),
            angular_orders=tuple(int(x) for x in angular_orders),
            contrast=float(contrast),
            radius_floor=float(radius_floor),
            channels=int(channels),
            init_std=float(init_std),
        )

        k = len(self.config.radial_frequencies)
        n = len(self.config.angular_orders)
        # Last dimension stores cosine/sine coefficients [a, b].
        self.coeff = nn.Parameter(torch.empty(channels, k, n, 2))
        nn.init.normal_(self.coeff, mean=0.0, std=init_std)

    @property
    def reference_size(self) -> int:
        return self.config.reference_size

    def _normalized_coeff(self) -> Tensor:
        # Prevent the trivial all-zero trigger while preserving the direction
        # learned by the optimizer. RMS normalization is smooth everywhere
        # except at the impossible exact all-zero initialization; epsilon handles it.
        rms = self.coeff.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        return self.coeff / (rms + 1e-6)

    def steered_coeff(self, scale_ratio: float | Tensor) -> Tensor:
        """Return coefficients after analytic scale steering.

        If phi = omega log(r) + n theta and delta = omega log(s), then
        q(r/s) = a cos(phi-delta) + b sin(phi-delta), which is represented by
            a_s = a cos(delta) - b sin(delta)
            b_s = a sin(delta) + b cos(delta).
        """
        device, dtype = self.coeff.device, self.coeff.dtype
        s = torch.as_tensor(scale_ratio, device=device, dtype=dtype)
        if torch.any(s <= 0):
            raise ValueError("scale_ratio must be positive")

        omega = torch.as_tensor(
            self.config.radial_frequencies, device=device, dtype=dtype
        )
        delta = omega * torch.log(s)
        cos_d = torch.cos(delta).view(1, -1, 1)
        sin_d = torch.sin(delta).view(1, -1, 1)

        c = self._normalized_coeff()
        a = c[..., 0]
        b = c[..., 1]
        a_s = a * cos_d - b * sin_d
        b_s = a * sin_d + b * cos_d
        return torch.stack([a_s, b_s], dim=-1)

    def _basis_grid(self, side: int) -> tuple[Tensor, Tensor]:
        if side < 2:
            raise ValueError("side must be >= 2")
        device, dtype = self.coeff.device, self.coeff.dtype
        axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        # Exact log-radial steering is defined for r>0. The tiny central disk is
        # regularized by clamping r to radius_floor; it is the only deliberate
        # discretization regularization in the renderer.
        radius_safe = radius.clamp_min(self.config.radius_floor)
        log_r = torch.log(radius_safe)
        theta = torch.atan2(yy, xx)
        return log_r, theta

    def render(self, side: int) -> Tensor:
        """Render an RGB trigger in [0, 1] with shape [3, side, side]."""
        log_r, theta = self._basis_grid(side)
        device, dtype = self.coeff.device, self.coeff.dtype
        coeff = self.steered_coeff(float(side) / float(self.reference_size))

        omega = torch.as_tensor(
            self.config.radial_frequencies, device=device, dtype=dtype
        )
        orders = torch.as_tensor(
            self.config.angular_orders, device=device, dtype=dtype
        )

        # [K, N, H, W]
        phase = (
            omega[:, None, None, None] * log_r[None, None, :, :]
            + orders[None, :, None, None] * theta[None, None, :, :]
        )
        cos_basis = torch.cos(phase)
        sin_basis = torch.sin(phase)

        a = coeff[..., 0][:, :, :, None, None]
        b = coeff[..., 1][:, :, :, None, None]
        signal = (a * cos_basis[None] + b * sin_basis[None]).sum(dim=(1, 2))
        signal = signal / math.sqrt(len(self.config.radial_frequencies) * len(self.config.angular_orders))

        # A bounded differentiable renderer. 0.5 corresponds to neutral gray.
        patch = 0.5 * (torch.tanh(self.config.contrast * signal) + 1.0)
        return patch.clamp(0.0, 1.0)

    @torch.no_grad()
    def render_numpy_bgr(self, side: int) -> np.ndarray:
        rgb = self.render(side).detach().cpu().permute(1, 2, 0).numpy()
        bgr = rgb[..., ::-1]
        return np.ascontiguousarray(bgr.astype(np.float32))

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)


def load_trigger(path: str | Path, device: str | torch.device = "cpu") -> ScaleSteerableTrigger:
    payload = torch.load(path, map_location=device)
    cfg = payload["config"]
    # torch serialization may turn tuples into lists; the constructor accepts both.
    trigger = ScaleSteerableTrigger(**cfg).to(device)
    trigger.load_state_dict(payload["state_dict"])
    return trigger
