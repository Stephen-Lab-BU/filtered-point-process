import numpy as np
import json
import os
import warnings
from .helpers import (
    TimeDomain,
    FrequencyDomain,
    create_frequency_domain,
    create_time_domain,
)
from .ParamSetter import ParamSetter, GlobalSeed
from .utils import nextpow2
from scipy.stats import norm


class ConditionalIntensityFunction(ParamSetter, GlobalSeed):
    def __init__(self, params=None, seed=None, config_file=None):
        super().set_params(config_file, params)
        super()._set_seed(seed)
        del seed

        # Global parameters #

        # Time_axis
        N = int(self.params["T"] * self.params["fs"])  # Number of samples
        self._time_axis = np.linspace(
            0, self.params["T"], N, endpoint=False
        )  # Time vector

        # Frequencies
        self.params["NFFT"] = self.__compute_NFFT()
        self.params["frequencies"] = self.__generate_frequency_vector(method="total")

        # Theoretical Power Spectrum
        self.PSD = self._spectrum()

        self.frequency_domain = create_frequency_domain(
            self.params["frequencies"], self.PSD
        )

    ######################### CORE FUNCTIONALITY #########################

    def _spectrum(self):
        """Compute the theoretical power spectral density."""

        if self.params["method"] == "gaussian":
            return self._compute_cox_cif_S()

        elif self.params["method"] == "Homogeneous_Poisson":
            return self._compute_homogeneous_cif_S()

        elif self.params["method"] == "AR(1)":
            return self._compute_ar1_cif_S()

        else:
            raise ValueError("Invalid method")

    def _simulate(self):
        """Generate a simulation of a time series process."""
        method_simulation = {
            "gaussian": self._simulate_gaussian,
            "Homogeneous_Poisson": self._simulate_homog_pois,
            "AR(1)": self._simulate_ar1,
        }

        simulation_func = method_simulation.get(self.params["method"])
        if not simulation_func:
            raise ValueError(f"Invalid method: {self.params['method']}")

        intensity = simulation_func()

        self.time_domain = create_time_domain(
            self._time_axis, intensity_realization=intensity
        )

    def __compute_NFFT(self):
        """Compute NFFT based on number of points in the time vector."""

        N = self.params["T"] * self.params["fs"]
        if nextpow2(N) < 100:
            return 10000
        else:
            return 100 * nextpow2(N)

    def __generate_frequency_vector(self, method="total"):
        """
        Generate a frequency vector for the theoretical power spectrum.

        Method can be either total (all negative and positive frequencies) or positive (which only returns positive frequencies up to Nyquist (i.e., fs // 2))

        Parameters:
        nfft (int): Number of FFT points.
        fs (float): Sampling frequency.

        Returns:
        np.ndarray: Frequency vector.

        Raises:
        ValueError: If the method is not 'total' or 'positive'.
        """
        if method == "total":
            # Generate the frequency vector
            frequencies = np.fft.fftfreq(self.params["NFFT"], d=1 / self.params["fs"])
            return np.abs(frequencies)
        elif method == "positive":
            frequencies = np.fft.fftfreq(self.params["NFFT"], d=1 / self.params["fs"])
            return frequencies[: self.params["NFFT"] // 2]
        else:
            # Raise an error if the method is not recognized
            raise ValueError("Invalid method. Choose 'total' or 'positive'.")

    ######################### HOMOGENEOUS POISSON PROCESS #########################

    def _simulate_homog_pois(self):
        """Simulate homogenous Poisson process."""

        homogenous_pois_cif_S = self._compute_homogeneous_cif_S()
        return np.real(self._rate_function(homogenous_pois_cif_S, self.params["fs"]))

    def _compute_homogeneous_cif_S(self):
        """Compute the theoretical power spectrum for the homogeneous Poisson process (HPP).
        There is no frequency dependence for a HPP and since the power
        spectrum of a conditional intensity function (or intensity function
        in this case) is capturing the variance in the rate, there is no power
        with the DC offset removed. Otherwise, all power is at 0. Here, we assumed
        a removed DC offset, like all other CIFs in the package."""

        return 0 * np.ones(self.params["frequencies"].shape)

    def _rate_function(self, S, fs):
        """Generate realizations of a homogeneous Poisson process in the time domain,
        with each realization being a horizontal line at lambda."""

        # Create an array with dimensions (number of simulations, length of t)
        rate_function_realizations = np.tile(
            self.params["lambda_0"] * np.ones_like(self._time_axis),
            (self.params["Nsims"], 1),
        )

        return rate_function_realizations

    ######################### COX PROCESS #########################

    ######################### GAUSSIAN PROCESS CIF #########################

    def _simulate_gaussian(self):
        """Simulate Gaussian process. This is the combined function for generating
        the theoretical power spectrum of the CIF (i.e., via _compute_cox_cif_S) and
        the time realization of the Gaussian (i.e., via _simulate_gaussian_process_approx_fd)
        """

        NFFT = self.__compute_NFFT()
        cox_cif_S = self._compute_cox_cif_S()
        intensity_realization = self._simulate_gaussian_process_approx_fd(
            cox_cif_S, len(self._time_axis), self.params["fs"]
        )
        return np.abs(np.real_if_close(intensity_realization[0]))

    def _compute_cox_cif_S(self):
        """Compute theoretical power spectrum for the Gaussian process."""

        cox_cif_S = self.params["peak_height"] * np.exp(
            -(
                (self.params["frequencies"].squeeze() - self.params["center_frequency"])
                ** 2
            )
            / (2 * self.params["peak_width"] ** 2)
        )
        cox_cif_S[0] = 0  # Enforce zero DC
        Mby2 = self.params["NFFT"] // 2
        cox_cif_S[Mby2 + 1 :] = np.flipud(cox_cif_S[1:Mby2])
        return cox_cif_S

    def _simulate_gaussian_process_approx_fd(self, S, N, fs):
        """Generate time domain realizations of a Gaussian process from the frequency
        defined Gaussian process in _compute_U_freqdomain. TO DO: Fix this function name
        to be more clear and change everywhere."""
        M = S.shape[0]
        U_freqdomain = self._compute_U_freqdomain(S, M)
        Y = np.real_if_close(np.sqrt(fs * M) * ifft(U_freqdomain, axis=0))
        cif_timedomain = Y[:N, :] + self.params["lambda_0"]
        cif_timedomain = np.real_if_close(cif_timedomain)
        cif_timedomain[cif_timedomain < 0] = 0  # Enforce non-negative rate
        return cif_timedomain, U_freqdomain

    def _compute_U_freqdomain(self, S, M):
        """
        _compute_U_freqdomain: Compute U in the frequency domain, which is a complex representation
        of a Gaussian process for simulation purposes.


        This function generates a frequency-domain representation of a Gaussian process that is guided
        by a given power spectrum, defined in S (i.e., via _compute_cox_cif_S).
        It creates a complex array, U_freqdomain, filled with random values
        drawn from a Gaussian distribution, scaled to match the power spectrum.


        Key points:
        - The real and imaginary parts of U_freqdomain for each frequency are derived from a Gaussian
        distribution, with variance determined by the power spectrum. This ensures that the resulting
        time-domain signal has the desired statistical properties.
        - The function handles the DC component and ensures symmetry between positive and negative
        frequencies to produce a real-valued time-domain signal.


        This function differs from _compute_cox_cif_S, which computes the theoretical power spectrum
        of the Gaussian process. While _compute_cox_cif_S defines the variance distribution over
        frequencies (e.g., a Gaussian-shaped curve around a certain frequency), _compute_U_freqdomain
        uses such a power spectrum to construct the actual frequency-domain representation of the
        Gaussian process. Essentially, _compute_cox_cif_S sets the theoretical framework for the
        power distribution, and _compute_U_freqdomain implements this framework to simulate the process.
        """

        U_freqdomain = np.zeros((M, self.params["Nsims"]), np.complex128)
        U_freqdomain[0, :] = np.sqrt(S[0]) * np.random.randn(self.params["Nsims"])
        Mby2 = M // 2
        std = np.sqrt(S[1:Mby2] / 2)
        U_freqdomain[1:Mby2, :] = std[:, None] * np.random.randn(
            Mby2 - 1, self.params["Nsims"]
        ) + 1j * std[:, None] * np.random.randn(Mby2 - 1, self.params["Nsims"])
        U_freqdomain[Mby2 + 1 :] = np.flipud(np.conj(U_freqdomain[1:Mby2, :]))
        if M % 2 == 0:
            U_freqdomain[Mby2, :] = np.sqrt(S[Mby2]) * np.random.randn(
                self.params["Nsims"]
            )
        return U_freqdomain

    ######################### AR(1) CIF #########################

    def _compute_ar1_cif_S(self):
        """Calculate the theoretical power spectrum for an AR(1) process."""

        omega = (2 * np.pi * self.params["frequencies"]) / (self.params["fs"])

        spectrum = (self.params["white_noise_variance"] ** 2) / (
            1 - 2 * self.params["phi_1"] * np.cos(omega) + self.params["phi_1"] ** 2
        )

        Mby2 = self.params["NFFT"] // 2
        spectrum[Mby2 + 1 :] = np.flipud(spectrum[1:Mby2])

        return spectrum

    def _compute_ar1_time(self, n_points, n_sims, c):
        time_series_array = np.zeros((n_sims, n_points))
        for sim in range(n_sims):
            white_noise = np.random.normal(
                0, self.params["white_noise_variance"], n_points
            )
            time_series = np.zeros(n_points)
            for t in range(1, n_points):
                time_series[t] = (
                    c + self.params["phi_1"] * time_series[t - 1] + white_noise[t]
                )
            time_series_array[sim, :] = time_series
            time_series_array = time_series_array + self.params["lambda_0"]
            time_series_array[time_series_array < 0] = 0
        return time_series_array

    def _simulate_ar1(self):
        """Simulate homogenous Poisson process."""

        sampling_frequency = 1 / (1 / self.params["fs"])
        t = np.arange(0, self.params["T"], (1 / self.params["fs"]))

        ar1_cif_S = self._compute_ar1_cif_S()
        ar1_cif_S[0] = 0  # Enforce zero DC

        # Calculate the variance and the desired mean
        var_Xt = self.params["white_noise_variance"] ** 2 / (
            1 - self.params["phi_1"] ** 2
        )
        mean_desired = 6 * np.sqrt(var_Xt)  # 6 standard deviations above zero
        c = (1 - self.params["phi_1"]) * mean_desired

        n_sims = self.params["Nsims"]
        return self._compute_ar1_time(n_points=len(self._time_axis), n_sims=n_sims, c=c)
