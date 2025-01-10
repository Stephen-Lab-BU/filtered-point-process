import numpy as np
from .base import FilterBase


class FastAPFilter(FilterBase):
    """
    Filter class for modeling Fast Action Potential (AP) processes.

    This filter represents the synaptic response associated with fast action potentials.
    It defines both the time-domain and frequency-domain kernels based on specified parameters.
    """

    def __init__(self, point_process, filter_params=None):
        """
        Initialize the FastAPFilter with specified parameters.

        Args:
            point_process (object): The point process or model instance to which the filter is applied.
            filter_params (dict, optional): Dictionary of parameters for configuring the filter.
                                            Supported keys include:
                                                - "k" (float): Scaling factor for the kernel. Defaults to 100.
                                                - "f0" (float): Center frequency in Hz. Defaults to 4000.
                                                - "theta" (float): Phase offset in radians. Defaults to π/4.
                                                - "sigma" (float): Standard deviation for the Gaussian envelope. Defaults to 0.00005.
                                                - "t0" (float): Time offset in seconds. Defaults to 0.0005.
        """
        super().__init__(point_process, filter_params=filter_params)

        self.k = self.filter_params.get("k", 100)
        self.f0 = self.filter_params.get("f0", 4000)
        self.theta = self.filter_params.get("theta", np.pi / 4)
        self.sigma = self.filter_params.get("sigma", 0.00005)
        self.t0 = self.filter_params.get("t0", 0.0005)

        self.compute_filter()

    def compute_filter(self):
        """
        Compute the time-domain and frequency-domain kernels for the Fast AP filter.

        This method calculates the time-domain kernel (`self._kernel_t`) using a Gabor function and
        computes the corresponding frequency-domain kernel (`self._kernel_f`)
        along with the power spectrum (`self._kernel_spectrum`).
        """
        fs = self.pp.cif.fs
        freqs = self.frequencies
        t = np.linspace(0, 1, int(fs))

        self.filter_params.setdefault("filter_time_vector", t)

        # Time-domain kernel
        self._kernel_t = (
            self.k
            * np.exp(-((t - self.t0) ** 2) / (2.0 * self.sigma**2))
            * np.cos(2 * np.pi * self.f0 * (t - self.t0) + self.theta)
        )

        # Frequency-domain kernel
        omega = 2 * np.pi * freqs
        exp_term = np.exp(
            -1j * self.theta
            - 2.0 * (self.f0**2) * (np.pi**2) * (self.sigma**2)
            - 1j * self.t0 * omega
            - 2.0 * self.f0 * np.pi * (self.sigma**2) * omega
            - (self.sigma**2) * (omega**2) / 2.0
        )
        cos_term = 1.0 + np.exp(
            2j * self.theta + 4.0 * self.f0 * np.pi * self.sigma**2 * omega
        )
        self._kernel_f = (
            self.k
            * exp_term
            * cos_term
            * np.sqrt(np.pi / 2.0)
            / np.sqrt(1.0 / self.sigma**2)
        )

        self._kernel_spectrum = np.abs(self._kernel_f) ** 2


class SlowAPFilter(FilterBase):
    """
    Filter class for modeling Slow Action Potential (AP) processes.

    This filter represents the synaptic response associated with slow action potentials.
    It defines both the time-domain and frequency-domain kernels based on specified parameters.
    """

    def __init__(self, point_process, filter_params=None):
        """
        Initialize the SlowAPFilter with specified parameters.

        Args:
            point_process (object): The point process or model instance to which the filter is applied.
            filter_params (dict, optional): Dictionary of parameters for configuring the filter.
                                            Supported keys include:
                                                - "k" (float): Scaling factor for the kernel. Defaults to 1.
                                                - "f0" (float): Center frequency in Hz. Defaults to 300.
                                                - "theta" (float): Phase offset in radians. Defaults to π/4.
                                                - "sigma" (float): Standard deviation for the Gaussian envelope. Defaults to 0.0005.
                                                - "t0" (float): Time offset in seconds. Defaults to 0.0015.
        """

        super().__init__(point_process, filter_params=filter_params)

        self.k = self.filter_params.get("k", 1)
        self.f0 = self.filter_params.get("f0", 300)
        self.theta = self.filter_params.get("theta", np.pi / 4)
        self.sigma = self.filter_params.get("sigma", 0.0005)
        self.t0 = self.filter_params.get("t0", 0.0015)

        self.compute_filter()

    def compute_filter(self):
        """
        Compute the time-domain and frequency-domain kernels for the Slow AP filter.

        This method calculates the time-domain kernel (`self._kernel_t`) using a Gaussian-modulated
        cosine function and computes the corresponding frequency-domain kernel (`self._kernel_f`)
        along with the power spectrum (`self._kernel_spectrum`).
        """
        fs = self.pp.cif.fs
        freqs = self.frequencies
        t = np.linspace(0, 1, int(fs * 1))

        self.filter_params.setdefault("filter_time_vector", t)

        # Time-domain kernel
        self._kernel_t = (
            self.k
            * np.exp(-((t - self.t0) ** 2) / (2.0 * self.sigma**2))
            * np.cos(2 * np.pi * self.f0 * (t - self.t0) + self.theta)
        )

        # Frequency-domain kernel
        omega = 2 * np.pi * freqs
        exp_term = np.exp(
            -1j * self.theta
            - 2.0 * (self.f0**2) * (np.pi**2) * (self.sigma**2)
            - 1j * self.t0 * omega
            - 2.0 * self.f0 * np.pi * (self.sigma**2) * omega
            - (self.sigma**2) * (omega**2) / 2.0
        )
        cos_term = 1.0 + np.exp(
            2j * self.theta + 4.0 * self.f0 * np.pi * self.sigma**2 * omega
        )
        self._kernel_f = (
            self.k
            * exp_term
            * cos_term
            * np.sqrt(np.pi / 2.0)
            / np.sqrt(1.0 / self.sigma**2)
        )

        self._kernel_spectrum = np.abs(self._kernel_f) ** 2
