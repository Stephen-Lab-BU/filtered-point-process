import numpy as np
import json
import os
import warnings
import math

from .cif import ConditionalIntensityFunction
from .helpers import (
    TimeDomain,
    FrequencyDomain,
    create_frequency_domain,
    create_time_domain,
)
from .ParamSetter import ParamSetter, GlobalSeed
from .utils import nextpow2


class PointProcess(ConditionalIntensityFunction, ParamSetter, GlobalSeed):
    def __init__(self, CIF=None, params=None, config_file=None, seed=None):
        # Initialize the parent class (ConditionalIntensityFunction)
        # super().__init__(params=params, config_file=config_file, seed=seed)
        super().set_params(config_file, params)
        super()._set_seed(seed)
        del seed

        # Store the ConditionalIntensityFunction instance
        self.cif = CIF

        # PP Parameters for naming clarity
        self.pp_params = params if params is not None else {}

        # Theoretical CIF Power Spectrum
        self.cif_PSD = self.cif._spectrum()

        self._set_process_type()

        # Theoretical PP Power Spectrum
        self.pp_PSD = self._spectrum()

        self.frequency_domain = create_frequency_domain(
            self.params["frequencies"], self.pp_PSD
        )

    def _set_process_type(self):
        method = self.pp_params["method"].lower()
        if method in ["ar(1)", "ar1", "gaussian"]:
            self._process_type = "cox"
        elif method == "homogeneous_poisson":
            self._process_type = "homog_pois"
        else:
            raise ValueError(f"Unknown method: {self.pp_params['method']}")

    def _generate_cox_spikes(self):
        """
        Simulate spikes in continuous time from the conditional intensity function (CIF)
        using the Lewis and Shedler (1979, algorithm 1) algorithm for an inhomogeneous Poisson process.

        Returns:
        np.ndarray: Array of spike times.
        """
        intensity = self.cif.time_domain.get_intensity_realization().squeeze()
        time_axis = self.cif.time_domain.get_time_axis()
        T = self.params["T"]
        lambda_max = np.max(intensity)

        # Initialize variables
        t = 0
        spike_times = []

        while t < T:
            # Generate u ~ uniform(0,1)
            u = np.random.uniform(0, 1)

            # Calculate w = -ln(u) / lambda_max
            w = -np.log(u) / lambda_max

            # Update t
            t = t + w

            if t >= T:
                break

            # Generate D ~ uniform(0,1)
            D = np.random.uniform(0, 1)

            # Accept t as a spike time with probability lambda(t) / lambda_max
            current_intensity = np.interp(t, time_axis, intensity)
            if D <= current_intensity / lambda_max:
                spike_times.append(t)

        return np.array(spike_times)

    def _generate_homog_pois_spikes(self, homog_pois_rate_value):
        """
        Generates spike trains based on the provided homogeneous Poisson process rate.

        Parameters:
        homog_pois_rate_value (float): The rate parameter (λ) for the homogeneous Poisson process.

        Returns:
        np.ndarray: Array of spike times.
        """
        T = self.params["T"]  # Total time
        lambda_rate = homog_pois_rate_value  # Poisson rate parameter

        # Initialize variables
        spike_times = []
        t = 0

        while True:
            # Generate u ~ uniform(0,1)
            u = np.random.uniform(0, 1)

            # Calculate w = -ln(u) / lambda_rate
            w = -np.log(u) / lambda_rate

            # Update t
            t = t + w

            # Check if the new time exceeds T
            if t > T:
                break

            # Record the spike time
            spike_times.append(t)

        return np.array(spike_times)

    def _extract_lambda(self):
        # Assuming cif_timedomain is an array-like structure
        cif_timedomain = (
            self.cif.time_domain.get_intensity_realization().squeeze().copy()
        )

        # Ensure non-negative values
        cif_timedomain[cif_timedomain < 0] = 0

        # Check if all values in cif_timedomain are the same
        if np.all(cif_timedomain == cif_timedomain[0]):
            lambda_value = cif_timedomain[0]
            return lambda_value
        else:
            raise ValueError(
                "Not all values in cif_timedomain are the same. Cannot extract a single lambda value."
            )

    def _simulate(self):
        """
        Simulates the point process based on the provided CIF.


        Returns:
        --------
        spikes : ndarray
            Simulated spike trains.
        cif_freqdomain : ndarray
            Frequency domain representation of the CIF.
        NFFT : int
            Number of FFT points.
        fs : float
            Sampling frequency.
        """
        if not (self._process_type == "cox" or self._process_type == "homog_pois"):
            raise ValueError(
                f"Process type '{self._process_type}' not currently implemented. Code base needs to be reworked to support this process type."
            )

        if self._process_type == "cox":
            N = int(math.floor(self.params["T"] / (1 / self.params["fs"])))
            cif_timedomain = (
                self.cif.time_domain.get_intensity_realization().squeeze().copy()
            )

            cif_timedomain[cif_timedomain < 0] = 0
            spikes = self._generate_cox_spikes()

            self.time_domain = create_time_domain(
                self.cif._time_axis, intensity_realization=cif_timedomain, events=spikes
            )

            return spikes

        elif self._process_type == "homog_pois":
            N = int(math.floor(self.params["T"] / (1 / self.params["fs"])))
            cif_timedomain = (
                self.cif.time_domain.get_intensity_realization().squeeze().copy()
            )

            cif_timedomain[cif_timedomain < 0] = 0  # should be redundant

            homog_pois_rate_value = self._extract_lambda()
            spikes = self._generate_homog_pois_spikes(homog_pois_rate_value)

            self.time_domain = create_time_domain(
                self.cif._time_axis, intensity_realization=cif_timedomain, events=spikes
            )

            return spikes

    def _spectrum(self):
        """
        Computes the power spectrum of the point process.
        """
        if not (self._process_type == "cox" or self._process_type == "homog_pois"):
            raise ValueError(
                f"Process type '{self._process_type}' not currently implemented. Code base needs to be reworked to support this process type."
            )
        if self._process_type == "cox" or self._process_type == "homog_pois":
            # if self.params["method"] == "gaussian":

            cif_power_spectrum = self.cif_PSD

            if np.isscalar(self.params["lambda_0"]):
                self.pp_power_spectrum = (
                    self.params["lambda_0"] * np.ones(cif_power_spectrum.shape)
                    + cif_power_spectrum
                )
            else:
                self.pp_power_spectrum = [
                    lambda0 * np.ones(cif_power_spectrum.shape) + cif_power_spectrum
                    for lambda0 in self.params["lambda_0"]
                ]

            return np.squeeze(self.pp_power_spectrum)
