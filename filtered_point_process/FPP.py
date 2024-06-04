from .filters import Filter
import numpy as np

class FilteredPointProcess(Filter):
    """Class to interact with the filtered point process."""

    def __init__(self, filter_1_model_1=None, filter_2_model_1=None, filter_1_model_2=None, filter_2_model_2=None, model_1=None, model_2=None, filter_1_config_file=None, filter_2_config_file=None):
        super().__init__(filter_1_model_1, filter_2_model_1, filter_1_model_2, filter_2_model_2, model_1, model_2, filter_1_config_file, filter_2_config_file)

    def get_individual_outputs(self):
        outputs = {}

        if hasattr(self, 'filter_1_instance_model_1') and self.filter_1_instance_model_1:
            outputs['model_1_filter_1'] = {
                "time_axis": self.filter_1_instance_model_1.kernel_time_axis,
                "kernel": self.filter_1_instance_model_1.kernel,
                "power_spectrum": self.filter_1_instance_model_1.kernel_spectrum,
                "frequencies": self.filter_1_instance_model_1.frequencies
            }

        if hasattr(self, 'filter_2_instance_model_1') and self.filter_2_instance_model_1:
            outputs['model_1_filter_2'] = {
                "time_axis": self.filter_2_instance_model_1.time_axis,
                "kernel": self.filter_2_instance_model_1.kernel,
                "power_spectrum": self.filter_2_instance_model_1.kernel_spectrum,
                "frequencies": self.filter_2_instance_model_1.frequencies
            }

        if hasattr(self, 'filter_1_instance_model_2') and self.filter_1_instance_model_2:
            outputs['model_2_filter_1'] = {
                "time_axis": self.filter_1_instance_model_2.kernel_time_axis,
                "kernel": self.filter_1_instance_model_2.kernel,
                "power_spectrum": self.filter_1_instance_model_2.kernel_spectrum,
                "frequencies": self.filter_1_instance_model_2.frequencies
            }

        if hasattr(self, 'filter_2_instance_model_2') and self.filter_2_instance_model_2:
            outputs['model_2_filter_2'] = {
                "time_axis": self.filter_2_instance_model_2.time_axis,
                "kernel": self.filter_2_instance_model_2.kernel,
                "power_spectrum": self.filter_2_instance_model_2.kernel_spectrum,
                "frequencies": self.filter_2_instance_model_2.frequencies
            }

        return outputs
    
    def perform_convolutions(self):
        """Perform convolutions with the kernels and spikes stored in the models."""
        results = {}

        if self.pp_1:
            spike_times_model_1 = self.pp_1.pp_events
            time_axis_model_1 = self.pp_1.pp_time_axis
            fs_model_1 = self.pp_1.params['fs']
            
            spike_train_model_1 = self._create_spike_train(spike_times_model_1, time_axis_model_1, fs_model_1)

            if hasattr(self, 'filter_1_instance_model_1') and self.filter_1_instance_model_1:
                kernel_1 = self.filter_1_instance_model_1.kernel
            else:
                kernel_1 = None
            if hasattr(self, 'filter_2_instance_model_1') and self.filter_2_instance_model_1:
                kernel_2 = self.filter_2_instance_model_1.kernel
            else:
                kernel_2 = None
            sim_PSPs_model_1, sim_LFP_model_1 = self._convolve_spikes_with_kernels(spike_train_model_1, kernel_1, kernel_2, fs_model_1)
            results['model_1'] = {
                "sim_PSPs": sim_PSPs_model_1,
                "sim_LFP": sim_LFP_model_1
            }

        if self.pp_2:
            spike_times_model_2 = self.pp_2.pp_events
            time_axis_model_2 = self.pp_2.pp_time_axis
            fs_model_2 = self.pp_2.params['fs']
            
            spike_train_model_2 = self._create_spike_train(spike_times_model_2, time_axis_model_2, fs_model_2)

            if hasattr(self, 'filter_1_instance_model_2') and self.filter_1_instance_model_2:
                kernel_1 = self.filter_1_instance_model_2.kernel
            else:
                kernel_1 = None
            if hasattr(self, 'filter_2_instance_model_2') and self.filter_2_instance_model_2:
                kernel_2 = self.filter_2_instance_model_2.kernel
            else:
                kernel_2 = None
            sim_PSPs_model_2, sim_LFP_model_2 = self._convolve_spikes_with_kernels(spike_train_model_2, kernel_1, kernel_2, fs_model_2)
            results['model_2'] = {
                "sim_PSPs": sim_PSPs_model_2,
                "sim_LFP": sim_LFP_model_2
            }

        return results

    def _create_spike_train(self, spike_times, time_axis, fs):
        """Create a spike train from the spike times and the time axis."""
        spike_train = np.zeros(len(time_axis))
        spike_indices = np.searchsorted(time_axis, spike_times)
        spike_train[spike_indices] = 1
        return spike_train

    def _convolve_spikes_with_kernels(self, spike_train, kernel_1, kernel_2, fs):
        """Helper function to perform convolutions with given kernels and spike train."""

        num_samples = len(spike_train)
        sim_PSPs = np.zeros(num_samples)
        sim_LFP = np.zeros(num_samples)

        # Perform convolution for each spike event
        sim_PSPs = np.convolve(spike_train, kernel_1, mode='full')[:num_samples] / fs

        if kernel_2 is not None:
            sim_LFP = np.convolve(sim_PSPs, kernel_2, mode='full')[:num_samples] / fs
        else:
            sim_LFP = sim_PSPs  

        return sim_PSPs, sim_LFP


    def _get_h_spectra(self):
        """Calculate the theoretical power spectrum of the filter."""
        spectra = {}
        if hasattr(self, 'filter_1_instance_model_1') and self.filter_1_instance_model_1:
            secondary_filter = self.filter_2_instance_model_1 if hasattr(self, 'filter_2_instance_model_1') else None
            spectra['model_1_filter_1'] = self._calculate_spectrum(self.filter_1_instance_model_1, secondary_filter)
        if hasattr(self, 'filter_2_instance_model_1') and self.filter_2_instance_model_1:
            spectra['model_1_filter_2'] = self._calculate_spectrum(self.filter_2_instance_model_1)
        if hasattr(self, 'filter_1_instance_model_2') and self.filter_1_instance_model_2:
            secondary_filter = self.filter_2_instance_model_2 if hasattr(self, 'filter_2_instance_model_2') else None
            spectra['model_2_filter_1'] = self._calculate_spectrum(self.filter_1_instance_model_2, secondary_filter)
        if hasattr(self, 'filter_2_instance_model_2') and self.filter_2_instance_model_2:
            spectra['model_2_filter_2'] = self._calculate_spectrum(self.filter_2_instance_model_2)
        return spectra


    def _calculate_spectrum(self, filter_instance, secondary_filter_instance=None):
        """Helper function to calculate power spectrum for a given filter instance."""
        S = filter_instance.pp.pp_PSD
        f = filter_instance.pp.params['frequencies']
        pp_power_spectrum = np.squeeze(np.array(S))
        h_power_spectrum = filter_instance.kernel_spectrum * (pp_power_spectrum / filter_instance.pp.params['fs']**2)
        
        if secondary_filter_instance is not None:
            h_power_spectrum *= secondary_filter_instance.kernel_spectrum
        
        return f, h_power_spectrum
