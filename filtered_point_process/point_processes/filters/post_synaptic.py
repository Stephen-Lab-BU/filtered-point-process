# post_synaptic.py
import numpy as np
from .base import FilterBase

class PSPFilter(FilterBase):
    """
    Generic post-synaptic potential (PSP) filter with double-exponential kernel.

    Shared behavior:
      - Scalar gain `h` multiplies the time-domain kernel; spectrum scales by h^2.
      - Kernel time axis matches CIF time axis via pp.cif.cif_time_axis (helper on CIFBase).
      - Frequency axis prefers pp.cif.cif_frequencies if present (falls back to FilterBase.frequencies).
    """
    DEFAULTS = {
        "tau_rise": 0.4e-3,
        "tau_decay": 4e-3,
        "h": 1.0,
    }

    def __init__(self, point_process, filter_params=None):
        super().__init__(point_process, filter_params=filter_params)
        for k, v in PSPFilter.DEFAULTS.items():
            self.filter_params.setdefault(k, v)
        self.compute_filter()

    # ---- Helpers ------------------------------------------------------------
    def _time_axis(self) -> np.ndarray:
        """
        Use CIF helper to get the data time axis (exact length match).
        Falls back to fs/T if needed.
        """
        fs = float(self.pp.cif.fs)
        # Preferred: CIFBase property (uses TimeDomain inside CIF)
        try:
            t = np.asarray(self.pp.cif.cif_time_axis).squeeze()
            if t.ndim != 1:
                raise ValueError("pp.cif.cif_time_axis must be 1D.")
            return t
        except Exception:
            # Fallback: build from fs and T
            if not hasattr(self.pp.cif, "T") or self.pp.cif.T is None:
                raise ValueError("No CIF time axis available and T is missing; cannot size kernel.")
            N = int(round(self.pp.cif.T * fs))
            return np.arange(N, dtype=float) / fs

    def _freq_axis(self) -> np.ndarray:
        """Prefer CIF frequency axis; else use FilterBase-provided frequencies."""
        try:
            f = np.asarray(self.pp.cif.cif_frequencies).squeeze()
            if f.ndim == 1 and f.size > 0:
                return f
        except Exception:
            pass
        return self.frequencies

    # ---- Core ---------------------------------------------------------------
    def compute_filter(self):
        fs = float(self.pp.cif.fs)
        t = self._time_axis()
        freqs = self._freq_axis()

        tau_rise = float(self.filter_params["tau_rise"])
        tau_decay = float(self.filter_params["tau_decay"])
        h = float(self.filter_params.get("h", 1.0))

        # Time-domain double-exponential (EPSP/IPSP) scaled by h
        self.filter_params["filter_time_vector"] = t  # keep for introspection
        self._kernel_t = h * (np.exp(-t / tau_decay) - np.exp(-t / tau_rise))

        # Frequency-domain transfer function (analytic), amplitude scales with h
        jw = 1j * 2 * np.pi * freqs
        kernel_f = (1.0 / (1.0 / tau_decay + jw)) - (1.0 / (1.0 / tau_rise + jw))
        self._kernel_f = h * kernel_f  # => power scales by h^2 automatically

        _kernel_fsym = self._create_symmetric_frequency_response(self._kernel_f)
        self._kernel_spectrum = np.abs(_kernel_fsym) ** 2  # includes h^2

class AMPAFilter(PSPFilter):
    """AMPA EPSP filter (excitatory)."""
    def __init__(self, point_process, filter_params=None):
        filter_params = dict(
            {**PSPFilter.DEFAULTS, "tau_decay": 4e-3, "tau_rise": 0.4e-3},
            **(filter_params or {}),
        )
        super().__init__(point_process, filter_params=filter_params)

class GABAFilter(PSPFilter):
    """GABA IPSP filter (inhibitory)."""
    def __init__(self, point_process, filter_params=None):
        filter_params = dict(
            {**PSPFilter.DEFAULTS, "tau_decay": 10e-3, "tau_rise": 0.4e-3},
            **(filter_params or {}),
        )
        super().__init__(point_process, filter_params=filter_params)
