from .cif import ConditionalIntensityFunction
from .pp import PointProcess
from .ParamSetter import ParamSetter, GlobalSeed

import numpy as np


class Filter(ParamSetter, GlobalSeed):
    VALID_FILTER_1_TYPES = ["AMPA", "GABA", "Fast_AP", "Slow_AP"]
    VALID_FILTER_2_TYPES = [None, "1/f"]

    def __init__(
        self,
        filter_1_model_1=None,
        filter_2_model_1=None,
        filter_1_model_2=None,
        filter_2_model_2=None,
        model_1=None,
        model_2=None,
        filter_1_config_file=None,
        filter_2_config_file=None,
    ):
        super().__init__()

        # Validate filter types
        self._validate_filter(
            filter_1_model_1, filter_2_model_1, filter_1_model_2, filter_2_model_2
        )

        # Global check for configuration files
        if filter_1_config_file or filter_2_config_file:
            raise ValueError(
                "Configuration files are not allowed. Please use the default parameters."
            )

        # Set filter types
        self.filter_1_model_1 = filter_1_model_1
        self.filter_2_model_1 = filter_2_model_1
        self.filter_1_model_2 = filter_1_model_2
        self.filter_2_model_2 = filter_2_model_2

        # Initialize models if provided
        self.model_1 = model_1
        self.model_2 = model_2

        if model_1 is not None:
            self.pp_1 = model_1
        else:
            self.pp_1 = None
            raise ValueError(
                "Model_1 was left as None, please specify a model you want to pass to the filters."
            )

        if model_2 is not None:
            self.pp_2 = model_2
        else:
            self.pp_2 = None

        # Set filter parameters
        self.filter_params = {}
        if filter_1_config_file:
            self._set_filter_params(filter_1_config_file)
            self._validate_config_filter(self.filter_params, self.VALID_FILTER_1_TYPES)
        if filter_2_config_file:
            self._set_filter_params(filter_2_config_file)
            self._validate_config_filter(self.filter_params, self.VALID_FILTER_2_TYPES)

        # Initialize filters
        if self.filter_1_model_1:
            self.filter_1_instance_model_1 = self.initialize_filter(
                self.filter_1_model_1, self.pp_1, self.filter_params
            )
        if self.filter_2_model_1:
            self.filter_2_instance_model_1 = self.initialize_filter(
                self.filter_2_model_1, self.pp_1, self.filter_params
            )
        if self.pp_2 is not None:
            if self.filter_1_model_2:
                self.filter_1_instance_model_2 = self.initialize_filter(
                    self.filter_1_model_2, self.pp_2, self.filter_params
                )
            if self.filter_2_model_2:
                self.filter_2_instance_model_2 = self.initialize_filter(
                    self.filter_2_model_2, self.pp_2, self.filter_params
                )

    def _validate_filter(
        self, filter_1_model_1, filter_2_model_1, filter_1_model_2, filter_2_model_2
    ):
        """Validate filter types."""
        filters = [
            filter_1_model_1,
            filter_2_model_1,
            filter_1_model_2,
            filter_2_model_2,
        ]
        valid_filters = self.VALID_FILTER_1_TYPES + self.VALID_FILTER_2_TYPES
        for f in filters:
            if f is not None and f not in valid_filters:
                raise ValueError(f"Filter must be one of {valid_filters}")

    def _validate_config_filter(self, filter_params, valid_types):
        """Validate filters in configuration."""
        if "method" in filter_params and filter_params["method"] not in valid_types:
            raise ValueError(
                f"Unsupported filter method in config, please provide one of the following {valid_types}."
            )

    def initialize_filter(self, filter_type, point_process, filter_params):
        """Initialize the appropriate filter based on the filter type."""
        if filter_type == "AMPA":
            return AMPAFilter(point_process=point_process, filter_params=filter_params)
        elif filter_type == "GABA":
            return GABAFilter(point_process=point_process, filter_params=filter_params)
        elif filter_type == "Fast_AP":
            return FastAPFilter(
                point_process=point_process, filter_params=filter_params
            )
        elif filter_type == "Slow_AP":
            return SlowAPFilter(
                point_process=point_process, filter_params=filter_params
            )
        elif filter_type == "1/f":
            return LeakyIntegratorFilter(
                point_process=point_process, filter_params=filter_params
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")


class FilterBase:
    """Base filter class for the post synaptic."""

    def __init__(
        self,
        point_process,
        tau_rise=None,
        tau_decay=None,
        filter_params=None,
        config_file=None,
        seed=None,
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
        self.filter_params["filter_1_time_vector"] = np.arange(
            0, 0.040, 1 / self.pp.params["fs"]
        )
        self._psc_t = np.exp(
            -self.filter_params["filter_1_time_vector"]
            / self.filter_params["tau_decay"]
        ) - np.exp(
            -self.filter_params["filter_1_time_vector"] / self.filter_params["tau_rise"]
        )

        # Normalize psc_t by its maximum value
        self._psc_t /= np.max(self._psc_t)

        self._psc_f = 1 / (
            1 / self.filter_params["tau_decay"] + 1j * 2 * np.pi * self.frequencies
        ) - 1 / (1 / self.filter_params["tau_rise"] + 1j * 2 * np.pi * self.frequencies)

        # Normalize psc_f by its maximum value
        self._psc_f /= np.max(np.abs(self._psc_f))

        self._psc_fsym = self._psc_f.copy()
        self._psc_fsym[int(np.floor(self.pp.params["NFFT"] / 2 + 1)) :] = np.flipud(
            np.conj(self._psc_fsym[1 : int(np.floor(self.pp.params["NFFT"] / 2))])
        )

        # Output theoretical spectrum of filter
        self._psc_S = np.abs(self._psc_fsym) ** 2

    @property
    def kernel_time_axis(self):
        """Return the time axis."""
        return self.filter_params["filter_1_time_vector"]

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

    def __init__(
        self, point_process, filter_params=None, tau_rise=None, tau_decay=None
    ):
        """Initialize filter with given point process and default time constants for AMPA."""
        tau_rise = (
            tau_rise
            if tau_rise is not None
            else filter_params.get("tau_rise", 0.4 / 1000)
        )
        tau_decay = (
            tau_decay
            if tau_decay is not None
            else filter_params.get("tau_decay", 2 / 1000)
        )
        super().__init__(point_process, tau_rise, tau_decay, filter_params)
        self.compute_psc()


class GABAFilter(FilterBase):
    """Filter class for GABA point processes."""

    def __init__(
        self, point_process, filter_params=None, tau_rise=None, tau_decay=None
    ):
        """Initialize filter with given point process and default time constants for GABA."""
        tau_rise = (
            tau_rise
            if tau_rise is not None
            else filter_params.get("tau_rise", 0.4 / 1000)
        )
        tau_decay = (
            tau_decay
            if tau_decay is not None
            else filter_params.get("tau_decay", 10 / 1000)
        )
        super().__init__(point_process, tau_rise, tau_decay, filter_params)
        self.compute_psc()


class LeakyIntegratorFilter:
    """Filter class for leaky integrator processes."""

    def __init__(
        self, point_process, A=1 / 0.1, filter_params=None, config_file=None, seed=None
    ):
        """Initialize filter with given point process and time constants."""
        if point_process is None or point_process.params is None:
            raise ValueError("Point process and its parameters must not be None.")

        self.pp = point_process

        self.filter_params = filter_params if filter_params is not None else {}
        self.filter_params["A"] = A
        self.filter_params["filter_2_time_vector"] = self.filter_params.get(
            "filter_2_time_vector", np.arange(0, 1, 1 / self.pp.params["fs"])
        )

        self.frequency = self.pp.params["frequencies"]
        self.NFFT = self.pp.params["NFFT"]
        self.compute_li()

    def compute_li(self):
        """Compute the leaky integrator response in both time and frequency domain."""
        self._li_t = np.exp(
            -self.filter_params["filter_2_time_vector"] * self.filter_params["A"]
        )
        self._li_f = 1 / (self.filter_params["A"] + 1j * 2 * np.pi * self.frequency)
        self._li_fsym = self._li_f.copy()
        self._li_fsym[int(np.floor(self.pp.params["NFFT"] / 2 + 1)) :] = np.flipud(
            np.conj(self._li_fsym[1 : int(np.floor(self.pp.params["NFFT"] / 2))])
        )

        self._li_fsym /= np.max(np.abs(self._li_fsym))

        self._li_S = np.abs(self._li_fsym) ** 2

    @property
    def time_axis(self):
        """Return the time axis."""
        return self.filter_params["filter_2_time_vector"]

    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._li_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._li_S

    @property
    def frequencies(self):
        """Return the frequency axis."""
        return self.pp.params["frequencies"]


class FastAPFilter:
    """Filter class for Fast Action Potential processes."""

    def __init__(self, point_process, filter_params=None, config_file=None, seed=None):
        """Initialize filter with given point process and default time constants for Fast AP."""
        self.filter_params = filter_params if filter_params is not None else {}
        self.pp = point_process
        self.k = 1
        self.f0 = 4000  # Center frequency in Hz
        self.theta = np.pi / 4  # Phase offset
        self.sigma = 0.00005  # Sigma in seconds
        self.t0 = 0.0005  # Center time in seconds
        self.compute_kernel()

    def compute_kernel(self):
        """Compute the kernel in both time and frequency domains."""
        t = np.linspace(
            -self.pp.params["T"] / 2,
            self.pp.params["T"] / 2,
            int(self.pp.params["fs"] * self.pp.params["T"]),
        )
        f = self.pp.params["frequencies"]

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
    def kernel_time_axis(self):
        """Return the time axis."""
        return np.linspace(
            -self.pp.params["T"] / 2,
            self.pp.params["T"] / 2,
            int(self.pp.params["fs"] * self.pp.params["T"]),
        )

    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._kernel_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._kernel_spectrum

    @property
    def frequencies(self):
        """Return the frequency axis."""
        return self.pp.params["frequencies"]


#### TO DO: Make the default parameters fill up the filter_params if it's empty and then pass that to the rest of the functions


class SlowAPFilter:
    """Filter class for Slow Action Potential processes."""

    def __init__(self, point_process, filter_params=None, config_file=None, seed=None):
        """Initialize filter with given point process and default time constants for Slow AP."""
        self.filter_params = filter_params if filter_params is not None else {}
        self.pp = point_process
        # Make filter_params dictionary with the followings:
        self.k = 1
        self.f0 = 300  # Center frequency in Hz
        self.theta = np.pi / 4  # Phase offset
        self.sigma = 0.0005  # Sigma in seconds
        self.t0 = 0.0015  # Center time in seconds
        self.compute_kernel()

    def compute_kernel(self):
        """Compute the kernel in both time and frequency domains."""
        t = np.linspace(
            -self.pp.params["T"] / 2,
            self.pp.params["T"] / 2,
            int(self.pp.params["fs"] * self.pp.params["T"]),
        )
        f = self.pp.params["frequencies"]

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
    def kernel_time_axis(self):
        """Return the time axis."""
        return np.linspace(
            -self.pp.params["T"] / 2,
            self.pp.params["T"] / 2,
            int(self.pp.params["fs"] * self.pp.params["T"]),
        )

    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._kernel_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._kernel_spectrum

    @property
    def frequencies(self):
        """Return the frequency axis."""
        return self.pp.params["frequencies"]
