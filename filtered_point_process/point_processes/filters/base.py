#base.py

import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d


class FilterBase(ABC):
    """
    Abstract base class for filters used in modeling point processes.

    Subclasses must implement the `compute_filter` method to define the filter's
    time-domain kernel (`self._kernel_t`), frequency-domain kernel (`self._kernel_f`),
    and power spectrum (`self._kernel_spectrum`).
    """

    def __init__(self, point_process, filter_params=None):
        """
        Initialize the FilterBase instance.

        Args:
            point_process (object): The point process or model instance associated with the filter.
            filter_params (dict, optional): Dictionary of parameters for configuring the filter.
                                            Defaults to an empty dictionary.
        """
        self.pp = point_process
        self.filter_params = filter_params if filter_params is not None else {}
        self._kernel_t = None
        self._kernel_f = None
        self._kernel_spectrum = None

    @abstractmethod
    def compute_filter(self):
        """
        Compute the filter's kernels and power spectrum.

        Subclasses must implement this method to:
            1. Define `self._kernel_t` (the time-domain kernel).
            2. Define `self._kernel_f` (the frequency-domain kernel, complex values).
            3. Define `self._kernel_spectrum` (power spectrum).
        """
        pass

    def _create_symmetric_frequency_response(self, freq_response):
        """
        Create a symmetric frequency response for negative frequencies.

        Args:
            freq_response (np.ndarray): The original frequency response array.

        Returns:
            np.ndarray: A symmetric frequency response array.
        """
        M = len(freq_response)
        freq_response_sym = freq_response.copy()
        Mby2 = M // 2

        if M % 2 == 0:
            freq_response_sym[Mby2:] = np.flipud(
                np.conj(freq_response_sym[1 : Mby2 + 1])
            )
        else:
            freq_response_sym[Mby2 + 1 :] = np.flipud(
                np.conj(freq_response_sym[1 : Mby2 + 1])
            )
        return freq_response_sym

    @property
    def frequencies(self):
        """
        Get the frequency vector associated with the point process.

        Returns:
            np.ndarray: Array of frequency values in Hz.
        """
        return self.pp.cif.frequencies

    @property
    def kernel_time_axis(self):
        """
        Get the time axis for the filter's time-domain kernel.

        Returns:
            np.ndarray or None: Array of time points in seconds, or None if not set.
        """
        return self.filter_params.get("filter_time_vector", None)

    @property
    def kernel(self):
        """
        Get the time-domain kernel of the filter.

        Returns:
            np.ndarray: Array representing the time-domain kernel.
        """
        return self._kernel_t

    @property
    def kernel_spectrum(self):
        """
        Get the power spectrum of the filter's frequency-domain kernel.

        Returns:
            np.ndarray: Array representing the power spectrum of the kernel.
        """
        return self._kernel_spectrum

    @property
    def kernel_density_not_squared(self):
        """
        Get the complex frequency-domain kernel of the filter.

        This property provides the frequency-domain representation without squaring,
        useful for certain analytical purposes.

        Returns:
            np.ndarray: Array representing the complex frequency-domain kernel.
        """
        return self._kernel_f
