#BasePP.py
from abc import ABC, abstractmethod
import numpy as np
from filtered_point_process.domains.frequency_domain import (
    create_frequency_domain,
    FrequencyDomain,
)


class BasePointProcess(ABC):
    """Abstract base class for point processes."""

    def __init__(self, CIF):
        """
        Initialize the BasePointProcess instance.

        Parameters
        ----------
        CIF : object
            The Conditional Intensity Function (CIF) instance associated with the point process.
        """
        self.cif = CIF
        self._set_process_type()
        self.time_domain = None
        self.frequency_domain = None

        if hasattr(self.cif, "cif_frequencies"):
            freqs = self.cif.cif_frequencies
        else:
            freqs = self.cif.get_frequencies()
 
        # 2) compute the “baseline” spectrum (λ₀ + CIF‐PSD)
        pp_psd = self._compute_spectrum()
 
        # 3) wrap it in your FrequencyDomain
        self.frequency_domain = create_frequency_domain(freqs, pp_psd)
        

    @abstractmethod
    def _set_process_type(self):
        """
        Set the process type based on parameters.

        This abstract method must be implemented by subclasses to define the specific type
        of point process being modeled.
        """
        pass

    @abstractmethod
    def simulate(self):
        """
        Simulate the point process.

        This abstract method must be implemented by subclasses to perform the simulation
        of the point process based on the defined CIF and other parameters.
        """
        pass

    def _generate_cox_spikes(self):
        """
        Generate Cox process spikes using the thinning algorithm.

        This method implements the thinning algorithm to generate spike times for a Cox
        process. It leverages the intensity realization from the CIF's time domain and
        performs stochastic thinning to produce the final spike times.

        Returns
        -------
        spikes : list or np.ndarray
            A list of arrays containing spike times for each simulation and process.
            The structure varies depending on whether the process is univariate or multivariate.
        """
        intensity = self.cif.time_domain.get_intensity_realization()
        time_axis = self.cif.time_domain.get_time_axis()
        T = self.cif.T  # Total time from the CIF

        assert isinstance(self.cif.seed, int), "Seed must be an integer."
        np_random = np.random.default_rng(self.cif.seed)

        time_axis = time_axis.flatten()
        if intensity.ndim > 1:
            intensity = intensity.squeeze()

        if (
            isinstance(intensity, np.ndarray)
            and intensity.ndim >= 2
            and intensity.shape[0] > 1
        ):
            # Multivariate case
            num_processes = intensity.shape[0]
            Nsims = intensity.shape[2] if intensity.ndim == 3 else 1
            spikes = []
            for sim in range(Nsims):
                spikes_sim = []
                for i in range(num_processes):
                    intensity_i = (
                        intensity[i, :, sim].flatten()
                        if Nsims > 1
                        else intensity[i, :].flatten()
                    )
                    lambda_max = np.max(intensity_i)
                    t = 0
                    spike_times_i = []
                    while t < T:
                        u = np_random.uniform(0, 1)
                        w = -np.log(u) / lambda_max
                        t = t + w
                        if t >= T:
                            break
                        D = np_random.uniform(0, 1)
                        current_intensity = np.interp(t, time_axis, intensity_i)
                        if D <= current_intensity / lambda_max:
                            spike_times_i.append(t)
                    spikes_sim.append(np.array(spike_times_i))
                spikes.append(spikes_sim)
            return spikes  # List of simulations, each containing list of spike times per process
        else:
            # Univariate case
            intensity = intensity.flatten()  # Ensure intensity is 1D
            lambda_max = np.max(intensity)
            t = 0
            spike_times = []
            while t < T:
                u = np_random.uniform(0, 1)
                w = -np.log(u) / lambda_max
                t = t + w
                if t >= T:
                    break
                D = np_random.uniform(0, 1)
                current_intensity = np.interp(t, time_axis, intensity)
                if D <= current_intensity / lambda_max:
                    spike_times.append(t)
            return np.array(spike_times)

    def _generate_homog_pois_spikes(self, lambda_rate):
        """
        Generate spikes from a homogeneous Poisson process.

        This method generates spike times for a homogeneous Poisson process (constant
        rate parameter `lambda_rate`) over the simulation interval.

        Parameters
        ----------
        lambda_rate : float or array-like
            The rate parameter(s) of the Poisson process. Can be a single float for univariate
            processes or a list/array of floats for multivariate processes.

        Returns
        -------
        spikes : list or np.ndarray
            A list of arrays containing spike times for each process. For multivariate processes,
            each array corresponds to a different subprocess.
        """
        T = self.cif.T  # Total time from the CIF
        np_random = np.random.RandomState(self.cif.seed)

        if isinstance(lambda_rate, (list, np.ndarray)):
            # Multivariate case
            spikes = []
            for rate in lambda_rate:
                spike_times = []
                t = 0
                while True:
                    u = np_random.uniform(0, 1)
                    w = -np.log(u) / rate
                    t = t + w
                    if t > T:
                        break
                    spike_times.append(t)
                spikes.append(np.array(spike_times))
            return spikes
        else:
            # Univariate case
            spike_times = []
            t = 0
            while True:
                u = np_random.uniform(0, 1)
                w = -np.log(u) / lambda_rate
                t = t + w
                if t > T:
                    break
                spike_times.append(t)
            return np.array(spike_times)

    def _extract_lambda(self):
        """
        Extract the lambda value(s) for homogeneous Poisson processes.

        This method retrieves the constant rate parameter(s) `lambda` from the CIF's
        time domain. It ensures that the intensity is constant over time, as required
        for homogeneous Poisson processes.

        Returns
        -------
        lambda_value : float or list of floats
            The extracted lambda value(s). Returns a single float for univariate processes
            or a list of floats for multivariate processes.

        Raises
        ------
        ValueError
            If the CIF intensity is not constant over time or has unexpected dimensions.
        """
        cif_timedomain = self.cif.time_domain.get_intensity_realization()

        # Ensure cif_timedomain is a numpy array
        cif_timedomain = np.asarray(cif_timedomain)

        # Ensure non-negative values
        cif_timedomain = np.clip(cif_timedomain, 0, None)

        if cif_timedomain.ndim == 0:
            return cif_timedomain.item()
        elif cif_timedomain.ndim == 1:
            # Univariate case
            if np.all(cif_timedomain == cif_timedomain[0]):
                return cif_timedomain[0]
            else:
                raise ValueError(
                    "Cannot extract single lambda value from variable CIF intensity."
                )
        elif cif_timedomain.ndim == 2:
            # Multivariate case
            num_processes = cif_timedomain.shape[0]
            lambdas = []
            for i in range(num_processes):
                intensity_i = cif_timedomain[i, :]
                if np.all(intensity_i == intensity_i[0]):
                    lambdas.append(intensity_i[0])
                else:
                    raise ValueError(
                        f"Cannot extract single lambda value from variable CIF intensity for process {i}"
                    )
            return lambdas
        else:
            raise ValueError(
                f"Unexpected number of dimensions in CIF intensity: {cif_timedomain.ndim}"
            )

    def _compute_spectrum(self):
        """
        Compute the power spectrum of the point process.

        This method calculates the power spectral density (PSD) of the point process based on
        the CIF's PSD. It handles both homogeneous Poisson processes and other types of CIFs,
        including multivariate scenarios.

        Returns
        -------
        pp_PSD : float or list of np.ndarray
            The computed power spectrum. Returns a single array for univariate processes or
            a list of arrays for multivariate processes.

        Raises
        ------
        ValueError
            If the CIF intensity has an unexpected number of dimensions or if required attributes
            are missing.
        """
        cif_PSD = self.cif.cif_PSD

        # Initialize flag for Homogeneous Poisson CIF
        is_homog_pois = False

        if "HomogeneousPoissonCIF" in type(self.cif).__name__:
            is_homog_pois = True

        elif (
            "MultivariateCIF" in type(self.cif).__name__
            and getattr(self.cif, "_cif_type_label", None) == "HomogeneousPoisson"
        ):
            is_homog_pois = True

        if is_homog_pois:
            pp_PSD = cif_PSD
            return pp_PSD

        if isinstance(cif_PSD, list):
            # Multivariate case
            pp_PSD = []
            for i in range(len(cif_PSD)):
                lambda_0 = self.cif.cifs[i].lambda_0
                pp_PSD_i = lambda_0 * np.ones_like(cif_PSD[i]) + cif_PSD[i]
                pp_PSD.append(pp_PSD_i)
            return pp_PSD
        else:
            # Univariate case
            lambda_0 = self.cif.lambda_0
            pp_PSD = lambda_0 * np.ones_like(cif_PSD) + cif_PSD
            return pp_PSD
