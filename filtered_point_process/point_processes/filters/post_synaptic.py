#post_synaptic.py

import numpy as np
from .base import FilterBase


class AMPAFilter(FilterBase):
    """
    Filter class for AMPA point processes (Post-Synaptic Potential).

    This filter models the excitatory post-synaptic potentials (EPSPs) mediated by AMPA receptors.
    It defines both the time-domain and frequency-domain kernels based on rise and decay time constants.
    """

    def __init__(self, point_process, filter_params=None):
        """
        Initialize the AMPAFilter with specified parameters.

        Args:
            point_process (object): The point process or model instance to which the filter is applied.
            filter_params (dict, optional): Dictionary of parameters for configuring the filter.
                                            Supported keys include:
                                                - "tau_rise" (float): Rise time constant in seconds. Defaults to 0.4e-3.
                                                - "tau_decay" (float): Decay time constant in seconds. Defaults to 4e-3.
        """
        super().__init__(point_process, filter_params=filter_params)

        if "tau_rise" not in self.filter_params:
            self.filter_params["tau_rise"] = 0.4e-3
        if "tau_decay" not in self.filter_params:
            self.filter_params["tau_decay"] = 4e-3

        self.compute_filter()

    def compute_filter(self):
        """
        Compute the time-domain and frequency-domain kernels for the AMPA filter.

        This method calculates the time-domain kernel (`self._kernel_t`) using a double exponential function
        representing the rise and decay of the post-synaptic potential. It then computes the corresponding
        frequency-domain kernel (`self._kernel_f`) and the power spectrum (`self._kernel_spectrum`).
        """
        fs = self.pp.cif.fs

        freqs = self.frequencies
            
        tau_rise = self.filter_params["tau_rise"]
        tau_decay = self.filter_params["tau_decay"]

        if self.pp.cif.simulate:
            T = self.pp.cif.T
            self.filter_params["filter_time_vector"] = np.linspace(0, 1, int(fs *1))

    
            self._kernel_t = np.exp(
                -self.filter_params["filter_time_vector"] / tau_decay
            ) - np.exp(-self.filter_params["filter_time_vector"] / tau_rise)

        self._kernel_f = 1.0 / (1.0 / tau_decay + 1j * 2 * np.pi * freqs) - 1.0 / (
            1.0 / tau_rise + 1j * 2 * np.pi * freqs
        )

        _kernel_fsym = self._create_symmetric_frequency_response(self._kernel_f)
        self._kernel_spectrum = np.abs(_kernel_fsym) ** 2


class GABAFilter(FilterBase):
    """
    Filter class for GABA point processes (Post-Synaptic Potential).

    This filter models the inhibitory post-synaptic potentials (IPSPs) mediated by GABA receptors.
    It defines both the time-domain and frequency-domain kernels based on rise and decay time constants.
    """

    def __init__(self, point_process, filter_params=None):
        """
        Initialize the GABAFilter with specified parameters.

        Args:
            point_process (object): The point process or model instance to which the filter is applied.
            filter_params (dict, optional): Dictionary of parameters for configuring the filter.
                                            Supported keys include:
                                                - "tau_rise" (float): Rise time constant in seconds. Defaults to 0.4e-3.
                                                - "tau_decay" (float): Decay time constant in seconds. Defaults to 10e-3.
        """
        super().__init__(point_process, filter_params=filter_params)

        if "tau_rise" not in self.filter_params:
            self.filter_params["tau_rise"] = 0.4e-3
        if "tau_decay" not in self.filter_params:
            self.filter_params["tau_decay"] = 10e-3

        self.compute_filter()

    def compute_filter(self):
        """
        Compute the time-domain and frequency-domain kernels for the GABA filter.

        This method calculates the time-domain kernel (`self._kernel_t`) using a double exponential function
        representing the rise and decay of the post-synaptic potential. It then computes the corresponding
        frequency-domain kernel (`self._kernel_f`) and the power spectrum (`self._kernel_spectrum`).
        """
        fs = self.pp.cif.fs
        freqs = self.frequencies

        

        tau_rise = self.filter_params["tau_rise"]
        tau_decay = self.filter_params["tau_decay"]

        if self.pp.cif.simulate:

            self.filter_params["filter_time_vector"] = np.linspace(0, 1, int(fs * 1))
            T = self.pp.cif.T
            # Time-domain
            self._kernel_t = np.exp(
                -self.filter_params["filter_time_vector"] / tau_decay
            ) - np.exp(-self.filter_params["filter_time_vector"] / tau_rise)

        # Frequency-domain
        self._kernel_f = 1.0 / (1.0 / tau_decay + 1j * 2 * np.pi * freqs) - 1.0 / (
            1.0 / tau_rise + 1j * 2 * np.pi * freqs
        )

        _kernel_fsym = self._create_symmetric_frequency_response(self._kernel_f)
        self._kernel_spectrum = np.abs(_kernel_fsym) ** 2
