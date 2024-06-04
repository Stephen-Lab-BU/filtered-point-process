from .cif import ConditionalIntensityFunction
from .pp import PointProcess
from .ParamSetter import ParamSetter, GlobalSeed

class PointProcessModel(ParamSetter, GlobalSeed):
    def __init__(self, params=None, config_file=None, seed=None):
        super().set_params(config_file, params)
        super()._set_seed(seed)
        self.cif = ConditionalIntensityFunction(params=self.params, seed = seed)
        self.pp = PointProcess(CIF = self.cif, params=self.params, seed = seed)

    def simulate_cif(self):
        self.cif._simulate()

    def simulate_pp(self):
        self.cif._simulate()
        self.pp._simulate()

    @property
    def cif_time_axis(self):
        return self.cif.time_domain.get_time_axis()

    @property
    def cif_realization(self):
        return self.cif.time_domain.get_intensity_realization().squeeze()

    @property
    def cif_frequencies(self):
        return self.cif.frequency_domain.get_frequencies()

    @property
    def cif_PSD(self):
        return self.cif.frequency_domain.get_PSD()

    @property
    def pp_time_axis(self):
        return self.pp.time_domain.get_time_axis()

    @property
    def pp_events(self):
        return self.pp.time_domain.get_events()

    @property
    def pp_frequencies(self):
        return self.pp.frequency_domain.get_frequencies()

    @property
    def pp_PSD(self):
        return self.pp.frequency_domain.get_PSD()