import numpy as np
from filtered_point_process.point_processes.BasePP import BasePointProcess
from filtered_point_process.cif.BaseCIF import CIFBase
from filtered_point_process.domains.time_domain import create_time_domain, TimeDomain
from filtered_point_process.domains.frequency_domain import (
    create_frequency_domain,
    FrequencyDomain,
)
from filtered_point_process.cif.MultivariateConstructor import MultivariateCIF
from filtered_point_process.cif.Spectral_Gaussian import GaussianCIF
from filtered_point_process.cif.HomogeneousPoisson import HomogeneousPoissonCIF
from filtered_point_process.cif.AR import ARCIF


class PointProcess(BasePointProcess):
    """
    Class to simulate a point process based on a Conditional Intensity Function (CIF).

    This class provides functionality to initialize the point process with a given CIF,
    determine the type of process (e.g., Cox, Homogeneous Poisson), and simulate spike
    events accordingly. It also handles the creation of time and frequency domain
    representations of the simulated data.
    """

    def __init__(self, CIF):
        """
        Initialize the PointProcess with a specified Conditional Intensity Function (CIF).

        Args:
            CIF (CIFBase): An instance of a CIF subclass that defines the intensity function.
        """
        super().__init__(CIF)
        self.spikes = None

    def _set_process_type(self):
        """
        Determine and set the type of point process based on the CIF.

        This method analyzes the CIF to identify whether the process supported type. It sets the internal
        `_process_type` attribute using that type.

        Raises:
            NotImplementedError: If a Multivariate AR CIF is provided, which is not yet implemented.
            ValueError: If an unsupported CIF type is encountered.
        """
        # Multivariate CIF
        if isinstance(self.cif, MultivariateCIF):
            if all(cif_type == "Gaussian" for cif_type in self.cif.cif_types):
                self._process_type = "cox"
            elif all(
                cif_type == "HomogeneousPoisson" for cif_type in self.cif.cif_types
            ):
                self._process_type = "homog_pois"
            elif any(cif_type == "AR" for cif_type in self.cif.cif_types):
                raise NotImplementedError("Multivariate AR CIF is not implemented yet.")
            else:
                raise ValueError("Unsupported CIF type in multivariate CIF.")
        else:
            # Univariate CIF
            method = self.cif.__class__.__name__
            if method == "GaussianCIF":
                self._process_type = "cox"
            elif method == "HomogeneousPoissonCIF":
                self._process_type = "homog_pois"
            elif method == "ARCIF":
                self._process_type = "cox"
            else:
                raise ValueError(f"Unknown or unsupported CIF type: {method}")

    def simulate(self):
        """
        Simulate the point process based on the initialized CIF.

        This method generates spike events by invoking the appropriate simulation
        method based on the determined process type (e.g., Cox, Homogeneous Poisson).
        After simulation, it constructs time and frequency domain representations
        of the spike data.

        Returns:
            list or np.ndarray: Simulated spike times. Returns a list of spike time arrays for
                                multivariate processes or a single array for univariate processes.

        Raises:
            ValueError: If the process type is not recognized or implemented.
            AttributeError: If frequency information is missing from the CIF or PointProcess.
        """
        if self._process_type == "cox":
            spikes = self._generate_cox_spikes()
        elif self._process_type == "homog_pois":
            lambda_value = self._extract_lambda()
            spikes = self._generate_homog_pois_spikes(lambda_value)
        else:
            raise ValueError(f"Process type '{self._process_type}' not implemented.")

        self.spikes = spikes

        # Create time domain object
        if isinstance(spikes, list):
            # Multivariate case
            events = spikes  # List of processes, each with spike times
            self.time_domain = create_time_domain(self.cif.time_axis, events=events)
        else:
            # Univariate case
            self.time_domain = create_time_domain(self.cif.time_axis, events=spikes)

        # Compute frequency domain data (TO DO: error handling remove in future versions)
        try:
            frequencies = self.cif.cif_frequencies
        except AttributeError:
            try:
                frequencies = self.cif.get_frequencies()
            except AttributeError:
                raise AttributeError(
                    "The CIF object does not have 'cif_frequencies' or 'get_frequencies' attributes."
                )

        pp_spectrum = self._compute_spectrum()
        self.frequency_domain = create_frequency_domain(frequencies, pp_spectrum)

        return spikes
