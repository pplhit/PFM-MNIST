from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class AngularSpectrumPropagator(torch.nn.Module):
    """Differentiable paraxial angular-spectrum propagation with padding."""

    def __init__(self, wavelength: float, dx: float, dz: float, pad_factor: int = 2) -> None:
        super().__init__()
        self.wavelength = float(wavelength)
        self.dx = float(dx)
        self.dz = float(dz)
        self.pad_factor = int(pad_factor)
        self._cache: dict[tuple, torch.Tensor] = {}

    def _transfer(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        hp, wp = h * self.pad_factor, w * self.pad_factor
        key = (hp, wp, str(device), dtype, self.wavelength, self.dx, self.dz)
        if key in self._cache:
            return self._cache[key]
        fy = torch.fft.fftfreq(hp, d=self.dx, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(wp, d=self.dx, device=device, dtype=dtype)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        kernel = torch.exp(-1j * math.pi * self.wavelength * self.dz * (fx_grid**2 + fy_grid**2))
        kernel = kernel[None, None]
        self._cache[key] = kernel
        return kernel

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(field):
            raise TypeError("AngularSpectrumPropagator expects a complex tensor.")
        _, _, h, w = field.shape
        if self.pad_factor > 1:
            hp, wp = h * self.pad_factor, w * self.pad_factor
            pt = (hp - h) // 2
            pb = hp - h - pt
            pl = (wp - w) // 2
            pr = wp - w - pl
            work = F.pad(field, (pl, pr, pt, pb))
        else:
            pt = pl = 0
            work = field
        kernel = self._transfer(h, w, field.device, field.real.dtype)
        out = torch.fft.ifft2(torch.fft.fft2(work, norm="ortho") * kernel, norm="ortho")
        if self.pad_factor > 1:
            out = out[..., pt : pt + h, pl : pl + w]
        return out


def phase_gradient(phi: torch.Tensor, dx: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local phase gradient without explicit phase unwrapping."""
    c = torch.exp(1j * phi)
    dc_dx = (torch.roll(c, -1, -1) - torch.roll(c, 1, -1)) / (2 * dx)
    dc_dy = (torch.roll(c, -1, -2) - torch.roll(c, 1, -2)) / (2 * dx)
    return torch.imag(dc_dx / (c + 1e-12)), torch.imag(dc_dy / (c + 1e-12))
