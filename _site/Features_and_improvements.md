# Features and Improvements

## Bug Fix To Do:

- [ ] Bug Fix 1: Description of bug fix 1.

Generalize simulation_params in model = Model(model_name = model_name, model_params = model_params, simulation_params = simulation_params) such that it can take values via []

Example: 

simulation_params = {
    'fs': 200,   # Sampling frequency in Hz
    'T': 5,      # Total time in seconds
    'simulate': True,
    'seed': 1  # For reproducibility
}

to

simulation_params = {
    'fs': [200],   # Sampling frequency in Hz
    'T': [5],      # Total time in seconds
    'simulate': True,
    'seed': [1]  # For reproducibility
}

model = Model(model_name, model_params, simulation_params)


## Test To Do

- [ ] Test 1: Unit test: model.cif.PSD == model.pp.frequency_domain.get_PSD()

- [ ] Test 2: Multivariate must have shared CIF, the following should turn an error

# Define model parameters
num_processes = 2
cif_types = ['Gaussian', 'Gaussian']
cif_params = [
    {
        'peak_height': [6],
        'center_frequency': [8],  # Hz
        'peak_width': [1.5],      # Hz
        'lambda_0': 20
    },
    {
        'peak_height': [4],
        'center_frequency': [12],  # Hz
        'peak_width': [2],         # Hz
        'lambda_0': 18
    }
]
weights = [0.5, -0.5]
simulation_params = {
    'fs': 200,
    'T': 5,
    'simulate': True,
    'seed': 5,
    'dependence': 'dependent',
    'weights': weights
}

# Create and simulate the model
model = Model('multivariate_gaussian', {
    'num_processes': num_processes,
    'cif_types': cif_types,
    'cif_params': cif_params,
}, simulation_params)

"
Process 0 parameters: {'peak_height': [6], 'center_frequency': [8], 'peak_width': [1.5]}
Process 1 parameters: {'peak_height': [4], 'center_frequency': [12], 'peak_width': [2]}

All processes must have identical spectral parameters (peak_height, center_frequency, peak_width) when 'dependence' is set to 'dependent'.
Please adjust your model parameters to ensure consistency."