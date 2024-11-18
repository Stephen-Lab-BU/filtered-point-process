from .cif import ConditionalIntensityFunction
from .pp import PointProcess
from .ParamSetter import ParamSetter, GlobalSeed
from scipy.integrate import simps


import numpy as np


class Filter(ParamSetter, GlobalSeed):
    VALID_FILTER_TYPES = ["AMPA", "GABA", "Fast_AP", "Slow_AP", "1/f", "Lorenzian"]

    def __init__(self, filters=None, model=None, filter_params=None):
        super().__init__()
        self.filters = filters
        self.model = model
        self.filter_params = filter_params if filter_params is not None else {}

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
            params = self.filter_params.get(filter_name, {})
            #print(f"Initializing filter '{filter_name}' with parameters: {params}")
            self.filter_instances[filter_name] = self.initialize_filter(
                filter_type, self.pp, params
            )

    def _validate_filters(self, filters):
        """Validate filter types."""
        for filter_type in filters.values():
            if filter_type not in self.VALID_FILTER_TYPES:
                raise ValueError(f"Filter must be one of {self.VALID_FILTER_TYPES}")

    def initialize_filter(self, filter_type, point_process, params):
        """Initialize the appropriate filter based on the filter type."""
        if filter_type == "AMPA":
            return AMPAFilter(point_process=point_process, filter_params=params)
        elif filter_type == "GABA":
            return GABAFilter(point_process=point_process, filter_params=params)
        elif filter_type == "Fast_AP":
            return FastAPFilter(point_process=point_process, filter_params=params)
        elif filter_type == "Slow_AP":
            return SlowAPFilter(point_process=point_process, filter_params=params)
        elif filter_type == "1/f":
            return LeakyIntegratorFilter(point_process=point_process, filter_params=params)
        elif filter_type == "Lorenzian":
            return LorenzianFilter(point_process=point_process, filter_params=params)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")


class FilterBase:
    """Base filter class for the post synaptic."""

    def __init__(self, point_process, tau_rise=None, tau_decay=None, filter_params=None):
        """Initialize filter with given point process and time constants."""
        self.pp = point_process
        self.filter_params = filter_params if filter_params is not None else {}
        self.filter_params["tau_rise"] = tau_rise if tau_rise is not None else self.filter_params.get("tau_rise")
        self.filter_params["tau_decay"] = tau_decay if tau_decay is not None else self.filter_params.get("tau_decay")

    def compute_psc(self):
        """Compute the post-synaptic current in both time and frequency domain."""
        #print(dir(self.pp))
        self.filter_params["filter_time_vector"] = np.arange(0, self.pp.params['T'], 1 / self.pp.params["fs"])
        self._psc_t = np.exp(-self.filter_params["filter_time_vector"] / self.filter_params["tau_decay"]) - np.exp(-self.filter_params["filter_time_vector"] / self.filter_params["tau_rise"])

        # Normalize psc_t by its maximum value
        #self._psc_t /= np.max(self._psc_t)

        self._psc_f = 1 / (1 / self.filter_params["tau_decay"] + 1j * 2 * np.pi * self.frequencies) - 1 / (1 / self.filter_params["tau_rise"] + 1j * 2 * np.pi * self.frequencies)

        # Normalize psc_f by its maximum value
        #self._psc_f /= np.abs(np.max(self._psc_t))**2

        self._psc_fsym = self._psc_f.copy()
        M = len(self._psc_fsym)
        Mby2 = M // 2

        if M % 2 == 0:
            self._psc_fsym[Mby2:] = np.flipud(np.conj(self._psc_fsym[1:Mby2 + 1]))
        else:
            self._psc_fsym[Mby2 + 1:] = np.flipud(np.conj(self._psc_fsym[1:Mby2 + 1]))

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
    
    @property
    def kernel_density_not_squared(self):
        """Return the magnitude of kernel axis."""
        return self._psc_fsym


class AMPAFilter(FilterBase):
    """Filter class for AMPA point processes."""

    def __init__(self, point_process, filter_params=None):
        """Initialize filter with given point process and default time constants for AMPA."""
        if filter_params and "tau_rise" in filter_params:
            tau_rise = filter_params["tau_rise"]
        else:
            tau_rise = 0.4 / 1000

        if filter_params and "tau_decay" in filter_params:
            tau_decay = filter_params["tau_decay"]
        else:
            tau_decay = 4 / 1000

        super().__init__(
            point_process,
            filter_params=filter_params,
            tau_rise=tau_rise,
            tau_decay=tau_decay,
        )
        self.compute_psc()


class GABAFilter(FilterBase):
    """Filter class for GABA point processes."""

    def __init__(self, point_process, filter_params=None):
        """Initialize filter with given point process and default time constants for GABA."""
        if filter_params and "tau_rise" in filter_params:
            tau_rise = filter_params["tau_rise"]
        else:
            tau_rise = 0.4 / 1000 # ms

        if filter_params and "tau_decay" in filter_params:
            tau_decay = filter_params["tau_decay"]
        else:
            tau_decay = 10 / 1000

        super().__init__(
            point_process,
            filter_params=filter_params,
            tau_rise=tau_rise,
            tau_decay=tau_decay,
        )
        self.compute_psc()

class LeakyIntegratorFilter(FilterBase):
    """Filter class for leaky integrator processes."""

    def __init__(self, point_process, filter_params=None):
        """Initialize filter with given point process and time constants."""
        filter_params = filter_params if filter_params is not None else {}
        
        if "A" not in filter_params:
            filter_params["A"] = 1 / 0.1
        
        if "filter_time_vector" not in filter_params:
            filter_params["filter_time_vector"] = np.arange(0, 1, 1 / point_process.params["fs"])
        
        super().__init__(point_process, filter_params=filter_params)
        self.compute_li()

    def compute_li(self):
        """Compute the leaky integrator response in both time and frequency domain."""
        self._li_t = np.exp(
            -self.filter_params["filter_time_vector"] * self.filter_params["A"]
        )

        #self._li_t /= np.max(self._li_t)
        self._li_f = 1 / (self.filter_params["A"] + 1j * 2 * np.pi * self.frequencies)
        self._li_fsym = self._li_f.copy()
        M = len(self._li_fsym)
        Mby2 = M // 2

        if M % 2 == 0:
            self._li_fsym[Mby2:] = np.flipud(
                np.conj(self._li_fsym[1:Mby2 + 1])
            )
        else:
            self._li_fsym[Mby2 + 1:] = np.flipud(
                np.conj(self._li_fsym[1:Mby2 + 1])
            )

        self._li_S = np.abs(self._li_fsym) ** 2

    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._li_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._li_S
    
    @property
    def kernel_density_not_squared(self):
        """Return the magnitude of kernel axis."""
        return self._li_fsym

class LorenzianFilter(FilterBase):
    """Filter class for Lorenzian processes with modified exponential decay."""

    def __init__(self, point_process, filter_params=None):
        """Initialize filter with given point process, time constants, and slope control."""
        filter_params = filter_params if filter_params is not None else {}
        
        # Default value for A (time constant)
        if "A" not in filter_params:
            filter_params["A"] = 1 / 0.1
        
        # Default value for alpha (slope control)
        if "alpha" not in filter_params:
            filter_params["alpha"] = 1.0
        
        # Default time vector
        if "filter_time_vector" not in filter_params:
            filter_params["filter_time_vector"] = np.arange(0, 1, 1 / point_process.params["fs"])
        
        super().__init__(point_process, filter_params=filter_params)
        self.compute_lorenzian()

    def compute_lorenzian(self):
        """Compute the Lorenzian filter response in both time and frequency domain."""
        A = self.filter_params["A"]
        alpha = self.filter_params["alpha"]
        time_vector = self.filter_params["filter_time_vector"]
        frequencies = self.frequencies

        # Compute time-domain response with modified exponential decay
        self._li_t = np.exp(-time_vector * A**alpha)

        # Compute frequency-domain response
        self._li_f = 1 / (A + 1j * 2 * np.pi * frequencies)**alpha
        self._li_fsym = self._li_f.copy()

        M = len(self._li_fsym)
        Mby2 = M // 2

        # Create symmetric frequency response
        if M % 2 == 0:
            self._li_fsym[Mby2:] = np.flipud(np.conj(self._li_fsym[1:Mby2 + 1]))
        else:
            self._li_fsym[Mby2 + 1:] = np.flipud(np.conj(self._li_fsym[1:Mby2 + 1]))

        # Compute power spectrum
        self._li_S = np.abs(self._li_fsym) ** 2

    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._li_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._li_S
    
    @property
    def kernel_density_not_squared(self):
        """Return the magnitude of kernel axis."""
        return self._li_fsym


class FastAPFilter(FilterBase):
    """Filter class for Fast Action Potential processes."""

    def __init__(self, point_process, filter_params=None):
        """Initialize filter with given point process and default time constants for Fast AP."""
        self.k = 100
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
        
        omega = 2 * np.pi * f

        # Exponential term (exp_term)
        exp_term = np.exp(-1j * self.theta 
                        - 2 * (self.f0**2) * (np.pi**2) * (self.sigma**2)
                        - 1j * self.t0 * omega 
                        - 2 * self.f0 * np.pi * self.sigma**2 * omega 
                        - (self.sigma**2) * (omega**2) / 2)

        # Cosine term (cos_term)
        cos_term = 1 + np.exp(2j * self.theta + 4 * self.f0 * np.pi * self.sigma**2 * omega)

        # Final analytical Fourier Transform (_kernel_f)
        self._kernel_f = self.k * exp_term * cos_term * np.sqrt(np.pi / 2) / np.sqrt(1 / self.sigma**2)

        # Spectrum calculation (magnitude squared of the frequency domain kernel)
        self._kernel_spectrum = np.abs(self._kernel_f) ** 2


    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._kernel_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._kernel_spectrum
    
    @property
    def kernel_density_not_squared(self):
        """Return the magnitude of kernel axis."""
        return self._kernel_f


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

        self.filter_params["filter_time_vector"] = self.filter_params.get(
            "filter_time_vector", t
        )

        self._kernel_t = (
            self.k
            * np.exp(-((t - self.t0) ** 2) / (2 * self.sigma**2))
            * np.cos(2 * np.pi * self.f0 * (t - self.t0) + self.theta)
        )
        
        # Angular frequency (omega)
        omega = 2 * np.pi * f

        # Exponential term (exp_term)
        exp_term = np.exp(-1j * self.theta 
                        - 2 * (self.f0**2) * (np.pi**2) * (self.sigma**2)
                        - 1j * self.t0 * omega 
                        - 2 * self.f0 * np.pi * self.sigma**2 * omega 
                        - (self.sigma**2) * (omega**2) / 2)

        # Cosine term (cos_term)
        cos_term = 1 + np.exp(2j * self.theta + 4 * self.f0 * np.pi * self.sigma**2 * omega)

        # Final analytical Fourier Transform (_kernel_f)
        self._kernel_f = self.k * exp_term * cos_term * np.sqrt(np.pi / 2) / np.sqrt(1 / self.sigma**2)

        # Spectrum calculation (magnitude squared of the frequency domain kernel)
        self._kernel_spectrum = np.abs(self._kernel_f) ** 2

    @property
    def kernel(self):
        """Return the time domain of the kernel."""
        return self._kernel_t

    @property
    def kernel_spectrum(self):
        """Return the frequency domain (power spectrum) of the kernel."""
        return self._kernel_spectrum
    
    @property
    def kernel_density_not_squared(self):
        """Return the magnitude of kernel axis."""
        return self._kernel_f
