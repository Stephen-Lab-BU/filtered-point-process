import pytest
import numpy as np
from unittest.mock import MagicMock


from filtered_point_process.point_processes.filters import (
    Filter,
    FilterBase,
    AMPAFilter,
    GABAFilter,
    FastAPFilter,
    SlowAPFilter,
    LeakyIntegratorFilter,
    LorenzianFilter,
)


@pytest.fixture
def mock_point_process():
    """
    Create a mock point-process object with essential attributes for testing.

    This fixture simulates a point-process-like object with minimal attributes required
    for filter operations, including sampling frequency, duration, and frequency bins.

    Attributes:
        cif.fs (int): Sampling frequency in Hz.
        cif.T (float): Total duration in seconds.
        cif.frequencies (numpy.ndarray): Array of frequency bins up to 500 Hz.

    Returns:
        MagicMock: A mock point-process object with predefined attributes.
    """
    mock_pp = MagicMock()
    mock_pp.cif.fs = 1000  # 1 kHz sampling
    mock_pp.cif.T = 1  # 1 second total
    mock_pp.cif.frequencies = np.linspace(0, 500, 501)  # up to 500 Hz
    return mock_pp


def test_filterbase_is_abstract(mock_point_process):
    """
    Ensure that FilterBase is an abstract class and cannot be instantiated directly.

    This test verifies that attempting to create an instance of FilterBase raises a
    TypeError, enforcing its role as an abstract base class.

    Parameters:
        mock_point_process (MagicMock): A mock point-process object for instantiation.

    Raises:
        TypeError: If FilterBase can be instantiated directly.
    """
    with pytest.raises(TypeError):
        _ = FilterBase(mock_point_process)


# ----------------------------
# Test each named filter
# ----------------------------


@pytest.mark.parametrize("FilterClass", [AMPAFilter, GABAFilter])
def test_post_synaptic_filters(mock_point_process, FilterClass):
    """
    Validate the function of post-synaptic filter kernels for AMPA and GABA filters.

    This test checks that the time-domain and frequency-domain kernels of AMPAFilter
    and GABAFilter are correctly generated, have appropriate shapes, and contain no
    invalid values such as NaNs or infinities.

    Parameters:
        mock_point_process (MagicMock): A mock point-process object for filter instantiation.
        FilterClass (class): The filter class to be tested (AMPAFilter or GABAFilter).

    Asserts:
        - Kernel and spectrum are not None.
        - Kernel length matches the product of sampling frequency and duration.
        - Spectrum length matches the number of frequency bins.
        - No NaN or infinite values are present in the kernel or spectrum.
    """
    f_inst = FilterClass(point_process=mock_point_process, filter_params={})

    # Basic checks
    assert f_inst.kernel is not None, "Time-domain kernel is None."
    assert f_inst.kernel_spectrum is not None, "Spectrum is None."
    assert len(f_inst.kernel) == int(
        mock_point_process.cif.fs * mock_point_process.cif.T
    ), "Time-domain kernel length does not match fs*T."
    assert len(f_inst.kernel_spectrum) == len(
        mock_point_process.cif.frequencies
    ), "Spectrum length does not match number of frequency bins."

    # Check for NaNs or infinities
    assert not np.isnan(f_inst.kernel).any(), "Time-domain kernel contains NaNs."
    assert not np.isnan(f_inst.kernel_spectrum).any(), "Spectrum contains NaNs."
    assert not np.isinf(f_inst.kernel).any(), "Time-domain kernel contains Infs."
    assert not np.isinf(f_inst.kernel_spectrum).any(), "Spectrum contains Infs."


@pytest.mark.parametrize("FilterClass", [FastAPFilter, SlowAPFilter])
def test_action_potential_filters(mock_point_process, FilterClass):
    """
    Assess the function of action potential filter kernels for FastAP and SlowAP filters.

    This test verifies that FastAPFilter and SlowAPFilter generate valid time-domain and
    frequency-domain kernels with correct shapes and without any NaN or infinite values.

    Parameters:
        mock_point_process (MagicMock): A mock point-process object for filter instantiation.
        FilterClass (class): The filter class to be tested (FastAPFilter or SlowAPFilter).

    Asserts:
        - Kernel and spectrum are not None.
        - Kernel length matches the product of sampling frequency and duration.
        - Spectrum length matches the number of frequency bins.
        - No NaN or infinite values are present in the kernel or spectrum.
    """

    f_params = {"k": 50, "f0": 2000, "theta": np.pi / 6, "sigma": 1e-4, "t0": 5e-4}
    f_inst = FilterClass(point_process=mock_point_process, filter_params=f_params)

    # Basic shape checks
    assert f_inst.kernel is not None
    assert f_inst.kernel_spectrum is not None
    assert len(f_inst.kernel) == int(
        mock_point_process.cif.fs * mock_point_process.cif.T
    )
    assert len(f_inst.kernel_spectrum) == len(mock_point_process.cif.frequencies)

    # No NaN or Inf
    assert not np.isnan(f_inst.kernel).any()
    assert not np.isnan(f_inst.kernel_spectrum).any()
    assert not np.isinf(f_inst.kernel).any()
    assert not np.isinf(f_inst.kernel_spectrum).any()


@pytest.mark.parametrize("FilterClass", [LeakyIntegratorFilter, LorenzianFilter])
def test_other_filters(mock_point_process, FilterClass):
    """
    Verify the functionality of "other" filters such as LeakyIntegrator and Lorenzian.

    This test ensures that LeakyIntegratorFilter and LorenzianFilter generate valid
    kernels with appropriate shapes and without any NaN or infinite values. It accommodates
    filters that may define their own time vectors by not enforcing a strict kernel length.

    Parameters:
        mock_point_process (MagicMock): A mock point-process object for filter instantiation.
        FilterClass (class): The filter class to be tested (LeakyIntegratorFilter or LorenzianFilter).

    Asserts:
        - Kernel and spectrum are not None.
        - Kernel length is greater than zero.
        - Spectrum length matches the number of frequency bins.
        - No NaN or infinite values are present in the kernel or spectrum.
    """
    f_inst = FilterClass(point_process=mock_point_process, filter_params={})
    # The default param sets might differ, but let's just test shape & validity

    assert f_inst.kernel is not None
    assert f_inst.kernel_spectrum is not None
    # Because these often define their own "filter_time_vector" length,
    # we can't strictly expect fs*T. But let's ensure it's > 0
    assert len(f_inst.kernel) > 0, "Time-domain kernel is empty."
    assert len(f_inst.kernel_spectrum) == len(
        mock_point_process.cif.frequencies
    ), "Spectrum length mismatch."

    # Check for validity
    assert not np.isnan(f_inst.kernel).any()
    assert not np.isnan(f_inst.kernel_spectrum).any()
    assert not np.isinf(f_inst.kernel).any()
    assert not np.isinf(f_inst.kernel_spectrum).any()


# ----------------------------
# Test the factory/manager
# ----------------------------


def test_filter_manager_instantiates(mock_point_process):
    """
    Confirm that the Filter manager correctly instantiates specified filter types.

    This test checks that given a dictionary of filter specifications, the Filter manager
    creates instances of the appropriate filter classes and stores them correctly.

    Parameters:
        mock_point_process (MagicMock): A mock point-process object for filter manager instantiation.

    Asserts:
        - The 'syn_exc' filter is an instance of AMPAFilter.
        - The 'syn_inh' filter is an instance of GABAFilter.
        - The 'li' filter is an instance of LeakyIntegratorFilter.
    """
    filters_dict = {"syn_exc": "AMPA", "syn_inh": "GABA", "li": "1/f"}
    from filtered_point_process.point_processes.filters import (
        Filter,
    )

    # Instantiate
    fman = Filter(filters=filters_dict, model=mock_point_process)

    # Check that we got the correct classes
    assert isinstance(fman.filter_instances["syn_exc"], AMPAFilter)
    assert isinstance(fman.filter_instances["syn_inh"], GABAFilter)
    assert isinstance(fman.filter_instances["li"], LeakyIntegratorFilter)


def test_filter_manager_invalid_filter_type(mock_point_process):
    """
    Ensure that the Filter manager raises an error for invalid filter types.

    This test verifies that providing an unrecognized filter type to the Filter manager
    results in a ValueError, maintaining the integrity of supported filter types.

    Parameters:
        mock_point_process (MagicMock): A mock point-process object for filter manager instantiation.

    Raises:
        ValueError: If an unknown or unsupported filter type is specified.
    """
    filters_dict = {"unknown": "NotARealFilter"}
    from filtered_point_process.point_processes.filters import Filter

    with pytest.raises(ValueError) as excinfo:
        _ = Filter(filters=filters_dict, model=mock_point_process)
    assert "Unknown filter type" in str(excinfo.value) or "Invalid filter type" in str(
        excinfo.value
    ), "Should raise error for invalid filter type."
