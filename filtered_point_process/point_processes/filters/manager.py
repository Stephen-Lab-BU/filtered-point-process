class Filter:
    """
    Factory and manager for initializing and managing filter instances.

    This class handles the creation of various filter types based on provided configurations.
    It validates the filter types, initializes the corresponding filter subclasses, and manages
    the collection of filter instances associated with a given model or point process.
    """

    VALID_FILTER_TYPES = ["AMPA", "GABA", "Fast_AP", "Slow_AP", "1/f", "Lorenzian"]

    def __init__(self, filters=None, model=None, filter_params=None):
        """
        Initialize the Filter manager with specified filters, model, and parameters.

        Args:
            filters (dict, optional): A dictionary mapping filter names to their types.
                                      Defaults to an empty dictionary.
            model (object, optional): The point process or model instance to which the filters
                                      will be applied. This parameter is required.
            filter_params (dict, optional): A dictionary of parameters for each filter.
                                            Defaults to an empty dictionary.

        Raises:
            ValueError: If the provided model is None.
            ValueError: If any of the specified filter types are invalid.
        """
        if filters is None:
            filters = {}
        self.filters = filters
        self.model = model
        self.filter_params = filter_params if filter_params else {}

        self._validate_filters(filters)

        if model is None:
            raise ValueError(
                "Model cannot be None; pass a valid point process or model."
            )

        self.pp = model  # The point process or model.pp

        self.filter_instances = {}
        for filter_name, filter_type in filters.items():
            fparams = self.filter_params.get(filter_name, {})
            inst = self.initialize_filter(filter_type, self.pp, fparams)
            self.filter_instances[filter_name] = inst

    def _validate_filters(self, filters):
        """
        Validate the provided filter types against the list of valid filter types.

        Args:
            filters (dict): A dictionary mapping filter names to their types.

        Raises:
            ValueError: If any filter type is not among the valid filter types.
        """

        for ftype in filters.values():
            if ftype not in self.VALID_FILTER_TYPES:
                raise ValueError(
                    f"Invalid filter type: {ftype}. Must be in {self.VALID_FILTER_TYPES}"
                )

    def initialize_filter(self, filter_type, point_process, params):
        """
        Initialize a specific filter instance based on the filter type.

        Args:
            filter_type (str): The type of filter to initialize. Must be one of the valid filter types.
            point_process (object): The point process or model instance associated with the filter.
            params (dict): A dictionary of parameters specific to the filter being initialized.

        Returns:
            FilterBase: An instance of the corresponding FilterBase subclass.

        Raises:
            ValueError: If the provided filter type is unknown or unsupported.
        """
        from .post_synaptic import AMPAFilter, GABAFilter
        from .action_potential import FastAPFilter, SlowAPFilter
        from .other import LeakyIntegratorFilter, LorenzianFilter

        if filter_type == "AMPA":
            return AMPAFilter(point_process, filter_params=params)
        elif filter_type == "GABA":
            return GABAFilter(point_process, filter_params=params)
        elif filter_type == "Fast_AP":
            return FastAPFilter(point_process, filter_params=params)
        elif filter_type == "Slow_AP":
            return SlowAPFilter(point_process, filter_params=params)
        elif filter_type == "1/f":
            return LeakyIntegratorFilter(point_process, filter_params=params)
        elif filter_type == "Lorenzian":
            return LorenzianFilter(point_process, filter_params=params)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
