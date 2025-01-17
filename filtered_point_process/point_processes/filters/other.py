import numpy as np
from .base import FilterBase


class LeakyIntegratorFilter(FilterBase):
    """
    Filter class for modeling leaky integrator processes (1/f-like structure).

    This filter represents a leaky integrator.
    It defines both the time-domain and frequency-domain kernels based on the leakage parameter.
    """

    def __init__(self, point_process, filter_params=None):
        """
        Initialize the LeakyIntegratorFilter with specified parameters.

        Args:
            point_process (object): The point process or model instance to which the filter is applied.
            filter_params (dict, optional): Dictionary of parameters for configuring the filter.
                                            Supported keys include:
                                                - "A" (float): Leakage parameter. Defaults to 10.0.
                                                - "filter_time_vector" (np.ndarray, optional): Time vector for the filter.
                                                                                             Defaults to an array from 0 to 1 second
                                                                                             with intervals based on the sampling frequency.
        """
        super().__init__(point_process, filter_params=filter_params)

        fs = self.pp.cif.fs
        T = self.pp.cif.T
        self.filter_params.setdefault("A", 1.0 / 0.1)
        self.filter_params.setdefault("filter_time_vector", np.linspace(0, 1, int(fs * 1)))

        self.compute_filter()

    def compute_filter(self):
        """
        Compute the time-domain and frequency-domain kernels for the leaky integrator filter.

        This method calculates the time-domain kernel (`self._kernel_t`) using an exponential decay function
        and computes the corresponding frequency-domain kernel (`self._kernel_f`) along with the power spectrum
        (`self._kernel_spectrum`).
        """
        A = self.filter_params["A"]
        t_vector = self.filter_params["filter_time_vector"]
        freqs = self.frequencies

        # Time-domain
        self._kernel_t = np.exp(-A * t_vector)

        self._kernel_t[0] = 0

        # Frequency-domain
        self._kernel_f = 1.0 / (A + 1j * 2 * np.pi * freqs)
        _kernel_fsym = self._create_symmetric_frequency_response(self._kernel_f)
        self._kernel_spectrum = np.abs(_kernel_fsym) ** 2


class LorenzianFilter(FilterBase):
    """
    Not fully tested!

    Filter class for modeling Lorentzian processes.

    This filter represents a Lorentzian process, characterized by its amplitude and shape parameters.
    It defines both the time-domain and frequency-domain kernels based on the specified parameters.
    """

    def __init__(self, point_process, filter_params=None):
        """
        Initialize the LorenzianFilter with specified parameters.

        Args:
            point_process (object): The point process or model instance to which the filter is applied.
            filter_params (dict, optional): Dictionary of parameters for configuring the filter.
                                            Supported keys include:
                                                - "A" (float): Scaling factor. Defaults to 10.0.
                                                - "alpha" (float): Shape parameter controlling the filter's width. Defaults to 1.0.
                                                - "filter_time_vector" (np.ndarray, optional): Time vector for the filter.
                                                                                             Defaults to an array from 0 to 1 second
                                                                                             with intervals based on the sampling frequency.
        """
        super().__init__(point_process, filter_params=filter_params)

        fs = self.pp.cif.fs
        self.filter_params.setdefault("A", 1.0 / 0.1)
        self.filter_params.setdefault("alpha", 1.0)
        self.filter_params.setdefault("filter_time_vector", np.arange(0, T, 1.0 / fs))

        self.compute_filter()

    def compute_filter(self):
        """
        Compute the time-domain and frequency-domain kernels for the Lorentzian filter.

        This method calculates the time-domain kernel (`self._kernel_t`) using an exponential decay function
        raised to the power of alpha and computes the corresponding frequency-domain kernel (`self._kernel_f`)
        along with the power spectrum (`self._kernel_spectrum`).
        """
        A = self.filter_params["A"]
        alpha = self.filter_params["alpha"]
        t_vector = self.filter_params["filter_time_vector"]
        freqs = self.frequencies

        # Time-domain
        self._kernel_t = np.exp(-(A**alpha) * t_vector)

        # Frequency-domain
        self._kernel_f = 1.0 / (A + 1j * 2 * np.pi * freqs) ** alpha
        _kernel_fsym = self._create_symmetric_frequency_response(self._kernel_f)
        self._kernel_spectrum = np.abs(_kernel_fsym) ** 2
