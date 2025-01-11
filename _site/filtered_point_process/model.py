from filtered_point_process.cif.Spectral_Gaussian import GaussianCIF
from filtered_point_process.cif.HomogeneousPoisson import HomogeneousPoissonCIF
from filtered_point_process.cif.AR import ARCIF
from filtered_point_process.cif.MultivariateConstructor import MultivariateCIF
from filtered_point_process.point_processes.pp import PointProcess


class Model:
    """
    Represents a point process model with its Conditional Intensity Function (CIF) and thinned continuous time spikes.
    """

    def __init__(self, model_name, model_params, simulation_params):
        """
        Initialize the Model with a specified CIF and simulation settings.

        Args:
            model_name (str): The name of the model to be used (e.g., 'gaussian', 'homogeneous_poisson').
            model_params (dict): Parameters specific to the chosen CIF model.
            simulation_params (dict): Parameters for simulation, including flags and simulation settings.
        """
        self.model_name = model_name.lower()
        self.model_params = model_params
        self.simulation_params = simulation_params

        self.cif = None
        self.pp = None
        self.spikes = None

        self._create_cif()

        if self.simulation_params.get("simulate", False) is True:
            self._simulate_process()

    def _create_cif(self):
        """
        Create the Conditional Intensity Function (CIF) based on the model name and parameters.

        This method initializes the appropriate CIF subclass instance corresponding to the specified model.

        Raises:
            ValueError: If the provided model name is unknown or unsupported.
        """
        sim_params = dict(self.simulation_params)

        if self.model_name == "gaussian":
            self.cif = GaussianCIF(**self.model_params, **sim_params)
        elif self.model_name == "homogeneous_poisson":
            self.cif = HomogeneousPoissonCIF(**self.model_params, **sim_params)
        elif self.model_name == "ar":
            self.cif = ARCIF(**self.model_params, **sim_params)
        elif self.model_name == "multivariate_gaussian":
            self.cif = MultivariateCIF(**self.model_params, **sim_params)
        elif self.model_name == "multivariate_homogeneous_poisson":
            self.cif = MultivariateCIF(**self.model_params, **sim_params)
        else:
            raise ValueError(f"Unknown or unsupported CIF type: {self.model_name}")

    def _simulate_process(self):
        """
        Simulate the point process based on the initialized CIF.

        This method generates spike events by simulating the point process using the associated CIF.
        The simulation results are stored within the model instance.
        """
        self.pp = PointProcess(self.cif)
        self.spikes = self.pp.simulate()

    @property
    def frequencies(self):
        """
        Retrieve the frequency vector associated with the CIF.

        Returns:
            np.ndarray: Array of frequency values in Hz.

        Raises:
            AttributeError: If frequencies are not found in the CIF or PointProcess.
        """
        if hasattr(self.pp, "frequency_domain"):
            return self.pp.frequency_domain.get_frequencies()
        elif hasattr(self.cif, "frequencies"):
            return self.cif.frequencies
        elif hasattr(self.cif, "cifs") and hasattr(self.cif.cifs[0], "frequencies"):
            return self.cif.cifs[0].frequencies
        else:
            raise AttributeError("Frequencies not found in CIF or PointProcess.")

    @property
    def time_axis(self):
        """
        Retrieve the time axis associated with the CIF.

        Returns:
            np.ndarray: Array representing the time points in seconds.

        Raises:
            AttributeError: If the time axis is not found in the CIF or PointProcess.
        """
        if hasattr(self.cif, "time_axis"):
            return self.cif.time_axis
        elif hasattr(self.cif, "time_domain") and hasattr(
            self.cif.time_domain, "get_time_axis"
        ):
            return self.cif.time_domain.get_time_axis()
        elif hasattr(self.cif, "cifs") and hasattr(self.cif.cifs[0], "time_axis"):
            return self.cif.cifs[0].time_axis
        else:
            raise AttributeError("Time axis not found in CIF.")
