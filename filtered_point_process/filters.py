from .cif import ConditionalIntensityFunction
from .pp import PointProcess
from .ParamSetter import ParamSetter, GlobalSeed

import numpy as np


class Filter(ParamSetter, GlobalSeed):
    VALID_FILTER_TYPES = ["AMPA", "GABA", "Fast_AP", "Slow_AP", "1/f"]

    def __init__(self, filters=None, model=None):
        super().__init__()
        self.filters = filters
        self.model = model

        # Validate filter types
        self._validate_filters(filters)

        if filters is None:
            filters = {}

        if model is not None:
            self.pp = model
        else:
            raise ValueError(
                "Model was left as None, please specify a model you want to pass to the filters."
            )

        # Initialize filters
        self.filter_instances = {}
        for filter_name, filter_type in filters.items():
            self.filter_instances[filter_name] = self.initialize_filter(
                filter_type, self.pp
            )

    def _validate_filters(self, filters):
        """Validate filter types."""
        for filter_type in filters.values():
            if filter_type not in self.VALID_FILTER_TYPES:
                raise ValueError(f"Filter must be one of {self.VALID_FILTER_TYPES}")

    def initialize_filter(self, filter_type, point_process):
        """Initialize the appropriate filter based on the filter type."""
        if filter_type == "AMPA":
            return AMPAFilter(point_process=point_process)
        elif filter_type == "GABA":
            return GABAFilter(point_process=point_process)
        elif filter_type == "Fast_AP":
            return FastAPFilter(point_process=point_process)
        elif filter_type == "Slow_AP":
            return SlowAPFilter(point_process=point_process)
        elif filter_type == "1/f":
            return LeakyIntegratorFilter(point_process=point_process)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")


class FilterBase:
    """Base filter class for the post synaptic."""

    def __init__(
        self, point_process, tau_rise=None, tau_decay=None, filter_params=None
    ):
        """Initialize filter with given point process and time constants."""
        self.pp = point_process
        self.filter_params = filter_params if filter_params is not None else {}
        self.filter_params["tau_rise"] = (
            tau_rise if tau_rise is not None else self.filter_params.get("tau_rise")
        )
        self.filter_params["tau_decay"] = (
            tau_decay if tau_decay is not None else self.filter_params.get("tau_decay")
        )

    def compute_psc(self):
        """Compute the post-synaptic current in both time and frequency domain."""
        self.filter_params["filter_time_vector"] = np.arange(
            0, 0.040, 1 / self.pp.params["fs"]
        )
        self._psc_t = np.exp(
            -self.filter_params["filter_time_vector"] / self.filter_params["tau_decay"]
        ) - np.exp(
            -self.filter_params["filter_time_vector"] / self.filter_params["tau_rise"]
        )

        # Normalize psc_t by its maximum value
        #self._psc_t /= np.max(self._psc_t)

        self._psc_f = 1 / (
            1 / self.filter_params["tau_decay"] + 1j * 2 * np.pi * self.frequencies
        ) - 1 / (1 / self.filter_params["tau_rise"] + 1j * 2 * np.pi * self.frequencies)

        # Normalize psc_f by its maximum value
        #self._psc_f /= np.max(np.abs(self._psc_f))

        self._psc_fsym = self._psc_f.copy()
        self._psc_fsym[int(np.floor(self.pp.params["NFFT"] / 2 + 1)) :] = np.flipud(
            np.conj(self._psc_fsym[1 : int(np.floor(self.pp.params["NFFT"] / 2))])
        )

        # Output theoretical spectrum of filter
        self._psc_S = np.abs(self._psc_fsym) ** 2

    @property
    def kernel_time_axis(self):
        """Return the time axis."""
        return self.filter_params["filter_time_vector"]

    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._psc_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._psc_S

    @property
    def frequencies(self):
        """Return the frequency axis."""
        return self.pp.params["frequencies"]


class AMPAFilter(FilterBase):
    """Filter class for AMPA point processes."""

    def __init__(self, point_process, filter_params=None):
        """Initialize filter with given point process and default time constants for AMPA."""
        super().__init__(
            point_process,
            filter_params=filter_params,
            tau_rise=0.4 / 1000,
            tau_decay=2 / 1000,
        )
        self.compute_psc()


class GABAFilter(FilterBase):
    """Filter class for GABA point processes."""

    def __init__(self, point_process, filter_params=None):
        """Initialize filter with given point process and default time constants for GABA."""
        super().__init__(
            point_process,
            filter_params=filter_params,
            tau_rise=0.4 / 1000,
            tau_decay=10 / 1000,
        )
        self.compute_psc()


class LeakyIntegratorFilter(FilterBase):
    """Filter class for leaky integrator processes."""

    def __init__(self, point_process, A=1 / 0.1, filter_params=None):
        """Initialize filter with given point process and time constants."""
        super().__init__(point_process, filter_params=filter_params)
        self.filter_params["A"] = A
        self.filter_params["filter_time_vector"] = self.filter_params.get(
            "filter_time_vector", np.arange(0, 1, 1 / self.pp.params["fs"])
        )
        self.compute_li()

    def compute_li(self):
        """Compute the leaky integrator response in both time and frequency domain."""
        self._li_t = np.exp(
            -self.filter_params["filter_time_vector"] * self.filter_params["A"]
        )
        self._li_f = 1 / (self.filter_params["A"] + 1j * 2 * np.pi * self.frequencies)
        self._li_fsym = self._li_f.copy()
        self._li_fsym[int(np.floor(self.pp.params["NFFT"] / 2 + 1)) :] = np.flipud(
            np.conj(self._li_fsym[1 : int(np.floor(self.pp.params["NFFT"] / 2))])
        )

        #self._li_fsym /= np.max(np.abs(self._li_fsym))

        self._li_S = np.abs(self._li_fsym) ** 2

    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._li_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._li_S


class FastAPFilter(FilterBase):
    """Filter class for Fast Action Potential processes."""

    def __init__(self, point_process, filter_params=None):
        """Initialize filter with given point process and default time constants for Fast AP."""
        self.k = 1
        self.f0 = 4000  # Center frequency in Hz
        self.theta = np.pi / 4  # Phase offset
        self.sigma = 0.00005  # Sigma in seconds
        self.t0 = 0.0005  # Center time in seconds
        super().__init__(point_process, filter_params=filter_params)
        self.compute_kernel()

    def compute_kernel(self):
        """Compute the kernel in both time and frequency domains."""
        t = np.linspace(0, 1, int(self.pp.params["fs"] * 1))
        f = self.pp.params["frequencies"]

        self.filter_params["filter_time_vector"] = self.filter_params.get(
            "filter_time_vector", t
        )

        self._kernel_t = (
            self.k
            * np.exp(-((t - self.t0) ** 2) / (2 * self.sigma**2))
            * np.cos(2 * np.pi * self.f0 * (t - self.t0) + self.theta)
        )
        self._kernel_f = self.k * np.exp(
            -2 * np.pi**2 * self.sigma**2 * (f - self.f0) ** 2
        ) * np.exp(1j * (self.theta - 2 * np.pi * f * self.t0)) + self.k * np.exp(
            -2 * np.pi**2 * self.sigma**2 * (f + self.f0) ** 2
        ) * np.exp(
            -1j * (self.theta + 2 * np.pi * f * self.t0)
        )
        self._kernel_spectrum = np.abs(self._kernel_f) ** 2

    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._kernel_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._kernel_spectrum


class SlowAPFilter(FilterBase):
    """Filter class for Slow Action Potential processes."""

    def __init__(self, point_process, filter_params=None):
        """Initialize filter with given point process and default time constants for Slow AP."""
        self.k = 1
        self.f0 = 300  # Center frequency in Hz
        self.theta = np.pi / 4  # Phase offset
        self.sigma = 0.0005  # Sigma in seconds
        self.t0 = 0.0015  # Center time in seconds
        super().__init__(point_process, filter_params=filter_params)
        self.compute_kernel()

    def compute_kernel(self):
        """Compute the kernel in both time and frequency domains."""
        t = np.linspace(0, 1, int(self.pp.params["fs"] * 1))
        f = self.pp.params["frequencies"]

        self._kernel_t = (
            self.k
            * np.exp(-((t - self.t0) ** 2) / (2 * self.sigma**2))
            * np.cos(2 * np.pi * self.f0 * (t - self.t0) + self.theta)
        )
        self.filter_params["filter_time_vector"] = self.filter_params.get(
            "filter_time_vector", t
        )
        self._kernel_f = self.k * np.exp(
            -2 * np.pi**2 * self.sigma**2 * (f - self.f0) ** 2
        ) * np.exp(1j * (self.theta - 2 * np.pi * f * self.t0)) + self.k * np.exp(
            -2 * np.pi**2 * self.sigma**2 * (f + self.f0) ** 2
        ) * np.exp(
            -1j * (self.theta + 2 * np.pi * f * self.t0)
        )
        self._kernel_spectrum = np.abs(self._kernel_f) ** 2

    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._kernel_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._kernel_spectrum
