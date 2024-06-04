import warnings
import os
import json
import numpy as np


class ParamSetter:
    params = None

    def _load_config(self, config_file):
        try:
            if os.path.isabs(config_file):
                config_path = config_file
            else:
                config_path = os.path.join(
                    os.path.dirname(__file__), "config", config_file
                )

            if not os.path.exists(config_path):
                warnings.warn(
                    f"Configuration file {config_path} not found. Using default parameters."
                )
                return None

            with open(config_path, "r") as f:
                params = json.load(f)
                if not isinstance(params, dict):
                    warnings.warn(
                        f"Configuration file {config_path} does not contain a valid dictionary. Using default parameters."
                    )
                    return None
                print(f"Configuration file {config_path} successfully loaded.")
                return {k: v["value"] for k, v in params.items() if "value" in v}
        except Exception as e:
            warnings.warn(
                f"Error loading configuration file {config_path}: {e}. Using default parameters."
            )
            return None

    def _default_params(self):
        return {
            "peak_height": 50000,
            "center_frequency": 1,
            "peak_width": 0.01,
            "fs": 10000,
            "T": 3,
            "Nsims": 1,
            "method": "gaussian",
            "lambda_0": [100],
        }

    def set_params(self, config_file=None, params=None):
        loaded_params = None
        if config_file:
            loaded_params = self._load_config(config_file)

        if loaded_params:
            self.__class__.params = loaded_params
        else:
            if params:
                self.__class__.params = params
            else:
                self.__class__.params = self._default_params()
                warnings.warn(
                    "Using default parameters as neither config_file nor user-defined parameters were provided. This is a Cox process with a Gaussian CIF by default."
                )

    def _set_filter_params(self, filter_config_file=None, filter_params=None):
        loaded_params = None
        if filter_config_file:
            loaded_params = self._load_config(filter_config_file)

        if loaded_params:
            self.__class__.filter_params = loaded_params
        else:
            if filter_params:
                self.__class__.params = filter_params
            else:
                warnings.warn(
                    "Default filter parameters are being used (see filters.py for parameter values for reporting purposes and equations.)"
                )

    def _default_params(self):
        return {
            "peak_height": 50000,
            "center_frequency": 1,
            "peak_width": 0.01,
            "fs": 10000,
            "T": 3,
            "Nsims": 1,
            "method": "gaussian",
            "lambda_0": [100],
        }


class GlobalSeed:
    global_seed = None

    def _set_seed(self, seed):
        """Set the random seed for reproducibility"""
        if seed is None:
            if self.__class__.global_seed is None:
                warnings.warn(
                    "No seed specified, simulations will yield different results each time.",
                    stacklevel=2,
                )
                np.random.seed(None)
            else:
                np.random.seed(self.__class__.global_seed)
                warnings.warn(
                    f"Using previously set global seed: {self.__class__.global_seed}",
                    stacklevel=2,
                )
        else:
            if not isinstance(seed, int):
                raise ValueError("Seed must be an integer.")
            np.random.seed(seed)
            self.__class__.global_seed = seed
            warnings.warn(
                f"Seed set globally to {self.__class__.global_seed} for reproducibility.",
                stacklevel=2,
            )
