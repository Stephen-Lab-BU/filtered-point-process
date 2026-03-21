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
        #t = self._time_axis()
        if self.filter_params.get("filter_time_vector", None) is not None:
            t = np.asarray(self.filter_params["filter_time_vector"], dtype=float).squeeze()
        else:
            t = self._time_axis()
        freqs = self._freq_axis()

        tau_rise = float(self.filter_params["tau_rise"])
        tau_decay = float(self.filter_params["tau_decay"])
        h = float(self.filter_params.get("h", 1.0))

        
        # Time-domain double-exponential (EPSP/IPSP) scaled by h
        self.filter_params["filter_time_vector"] = t
        self._kernel_t = h * (np.exp(-t / tau_decay) - np.exp(-t / tau_rise))

        # Build the frequency-domain kernel from the SAME sampled kernel used in simulation
        dt = 1.0 / fs
        #n_freq = len(freqs)
        if self.pp.cif.simulate and hasattr(self.pp.cif, "N"):
            n_freq = int(self.pp.cif.N)
            freqs = np.fft.fftshift(np.fft.fftfreq(n_freq, d=1.0 / fs))
        else:
            n_freq = len(freqs)

        # zero-pad / truncate onto the same FFT grid used by the CIF / point-process spectrum
        kernel_padded = np.zeros(n_freq, dtype=float)
        n_copy = min(len(self._kernel_t), n_freq)
        kernel_padded[:n_copy] = self._kernel_t[:n_copy]

        # FFT on the package's shifted frequency convention
        self._kernel_f = np.fft.fftshift(np.fft.fft(kernel_padded)) * dt 
        self._kernel_spectrum = np.abs(self._kernel_f) ** 2

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
