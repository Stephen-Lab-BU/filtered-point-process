# action_potential.py
import numpy as np
from .base import FilterBase

class APFilter(FilterBase):
    """
    Generic Action Potential (AP) filter with Gabor-like kernel.

    - Scalar gain `h` multiplies the time-domain kernel; power spectrum scales ~ h^2.
    - Kernel time axis equals CIF time axis (pp.cif.cif_time_axis), else fs/T fallback.
    - Frequency axis prefers CIF frequencies; else FilterBase.frequencies.
    """

    DEFAULTS = {
        "h": 1.0,         # scalar gain
        "f0": 1_000.0,    # center frequency [Hz]
        "theta": np.pi/4, # phase [rad]
        "sigma": 5e-4,    # Gaussian std [s]
        "t0": 1.0e-3,     # time offset [s]
    }

    def __init__(self, point_process, filter_params=None):
        super().__init__(point_process, filter_params=filter_params)
        # Fill defaults FIRST, then apply provided params, then compute
        for k, v in APFilter.DEFAULTS.items():
            self.filter_params.setdefault(k, v)
        if filter_params:
            self.filter_params.update(filter_params)
        self.compute_filter()

    # ----- public API to update params and force recompute -----
    def update_filter_params(self, params: dict | None = None):
        if params:
            self.filter_params.update(params)
        self.compute_filter()

    @property
    def h(self) -> float:
        return float(self.filter_params["h"])

    @h.setter
    def h(self, value: float):
        self.filter_params["h"] = float(value)
        self.compute_filter()

    # ----- axes helpers -----
    def _time_axis(self) -> np.ndarray:
        fs = float(self.pp.cif.fs)
        try:
            t = np.asarray(self.pp.cif.cif_time_axis).squeeze()
            if t.ndim != 1:
                raise ValueError("pp.cif.cif_time_axis must be 1D.")
            return t
        except Exception:
            if not hasattr(self.pp.cif, "T") or self.pp.cif.T is None:
                raise ValueError("No CIF time axis and T missing; cannot size AP kernel.")
            N = int(round(self.pp.cif.T * fs))
            if N <= 0:
                raise ValueError("Computed kernel length N <= 0. Check fs and T.")
            return np.arange(N, dtype=float) / fs

    def _freq_axis(self) -> np.ndarray:
        try:
            f = np.asarray(self.pp.cif.cif_frequencies).squeeze()
            if f.ndim == 1 and f.size > 0:
                return f
        except Exception:
            pass
        return self.frequencies

    # ----- core computation -----
    def compute_filter(self):
        t = self._time_axis()
        freqs = self._freq_axis().astype(float)

        h_gain = float(self.filter_params.get("h", 1.0))
        f0     = float(self.filter_params["f0"])
        theta  = float(self.filter_params["theta"])
        sigma  = float(self.filter_params["sigma"])
        t0     = float(self.filter_params["t0"])

        # --- Time domain: multiply by h ---
        self.filter_params["filter_time_vector"] = t
        g = np.exp(-((t - t0) ** 2) / (2.0 * sigma**2))
        c = np.cos(2 * np.pi * f0 * (t - t0) + theta)
        self._kernel_t = h_gain * g * c

        # --- Frequency domain: DO NOT multiply transfer function by h ---
        omega = 2 * np.pi * freqs
        exp_term = np.exp(
            -1j * theta
            - 2.0 * (f0**2) * (np.pi**2) * (sigma**2)
            - 1j * t0 * omega
            - 2.0 * f0 * np.pi * (sigma**2) * omega
            - (sigma**2) * (omega**2) / 2.0
        )
        cos_term = 1.0 + np.exp(2j * theta + 4.0 * f0 * np.pi * sigma**2 * omega)

        # Base complex frequency response (no h here)
        kernel_f_base = exp_term * cos_term * np.sqrt(np.pi / 2.0) * sigma
        self._kernel_f = kernel_f_base

        # --- Spectrum: multiply by h^2 ---
        _kernel_fsym = self._create_symmetric_frequency_response(self._kernel_f)
        self._kernel_spectrum = (np.abs(_kernel_fsym) ** 2) * (h_gain ** 2)

    # ----- raw (non-normalized) accessors -----
    @property
    def kernel(self):
        return self._kernel_t

    @property
    def kernel_time_axis(self):
        return self.filter_params["filter_time_vector"]

    @property
    def kernel_spectrum(self):
        return self._kernel_spectrum


class FastAPFilter(APFilter):
    def __init__(self, point_process, filter_params=None):
        defaults = {
            **APFilter.DEFAULTS,
            "h": 1.0,
            "f0": 4000.0,
            "theta": np.pi / 4,
            "sigma": 50e-6,  # 0.00005 s
            "t0": 0.0005,    # 0.5 ms
        }
        params = {**defaults, **(filter_params or {})}
        super().__init__(point_process, filter_params=params)


class SlowAPFilter(APFilter):
    def __init__(self, point_process, filter_params=None):
        defaults = {
            **APFilter.DEFAULTS,
            "h": 1.0,
            "f0": 300.0,
            "theta": np.pi / 4,
            "sigma": 0.0005,  # 0.5 ms
            "t0": 0.0015,     # 1.5 ms
        }
        params = {**defaults, **(filter_params or {})}
        super().__init__(point_process, filter_params=params)
