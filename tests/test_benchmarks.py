import pytest

from qiskit.providers.jobstatus import JobStatus
from qiskit_experiments.framework import AnalysisStatus
from qiskit_ibm_runtime.fake_provider import FakeFez

from qiskit_device_benchmarking.clops.clops_benchmark import (
    create_hardware_aware_circuit,
)
from qiskit_device_benchmarking.bench_code.bell import CHSHExperiment

# from qiskit_device_benchmarking.bench_code.dynamic_circuits_rb import DynamicCircuitsRB
# from qiskit_device_benchmarking.bench_code.mcm_rb_experiment import McmRB
from qiskit_device_benchmarking.bench_code.mrb.mirror_qv import MirrorQuantumVolume
from qiskit_device_benchmarking.bench_code.mrb.mirror_rb_experiment import MirrorRB
from qiskit_device_benchmarking.bench_code.prb.pur_rb import PurityRB


@pytest.fixture
def backend():
    return FakeFez()


def test_clops_hardware_aware_circuit(backend):
    qc, params = create_hardware_aware_circuit(
        width=100, layers=100, backend=backend, parameterized=False
    )

    assert qc.num_qubits == 100
    assert qc.depth() == 100
    assert not params

    qc, params = create_hardware_aware_circuit(
        width=100, layers=100, backend=backend, parameterized=True
    )

    assert qc.num_qubits == 100
    assert qc.depth() == 100
    assert params


# def test_mirror_pub(backend):
#     pub_options = MirrorPubOptions()
#     pub_options.num_qubits = 100
#     pub_options.target_num_2q_gates = 4986
#     pub_options.theta = 0
#     pub_options.path_strategy = "eplg_chain"
#
#     pubs = pub_options.get_pubs(backend)
#
#     for circuit, obs, params in pubs:
#         assert circuit.num_qubits == 100
#         assert circuit.depth() == 4986


def test_chsh_experiment(backend):
    exp = CHSHExperiment([0, 1])
    exp_data = exp.run(backend=backend).block_for_results()
    s = exp_data.analysis_results("S", dataframe=True).iloc[0]
    assert exp_data.job_status() == JobStatus.DONE
    assert exp_data.analysis_status() == AnalysisStatus.DONE
    assert s.value


# This test attempts to open matplotlib, which it should not be doing with this
# code. This code needs to be resolved for the test to be re-introduced.
#
# def test_bell_experiment(backend):
#     layered_coupling_map = [[(0, 1), (1, 0)]]
#     exp = BellExperiment(layered_coupling_map, backend=backend)
#     exp_data = exp.run(backend=backend).block_for_results()
#     hf = exp_data.analysis_results(dataframe=True)
#     assert exp_data.job_status() == JobStatus.DONE
#     assert exp_data.analysis_status() == AnalysisStatus.DONE
#
#     fidelity = hf.iloc[0].value.fidelity
#     assert fidelity


# ImportError causes tests to fail, error needs to be resolved for tests
# to be re-introduced.
#
# def test_dynamic_circuits_rb():
#     backend = FakeFractionalBackend()
#     exp = DynamicCircuitsRB(physical_qubits=backend.coupling_map.physical_qubits, backend=backend)
#     exp_data = exp.run(backend=backend).block_for_results()
#     assert exp_data.job_status() == JobStatus.DONE
#     assert exp_data.analysis_status() == AnalysisStatus.DONE
#
#
# def test_mcm_rb(backend):
#     exp = McmRB(
#         clif_qubit_sets=[(0, 1), (1, 0)],
#         meas_qubit_sets=[(0, 1), (1, 0)],
#         backend=backend
#     )
#     exp_data = exp.run(backend=backend).block_for_results()
#     assert exp_data.job_status() == JobStatus.DONE
#     assert exp_data.analysis_status() == AnalysisStatus.DONE


def test_mirror_qv(backend):
    exp = MirrorQuantumVolume(qubits=[0, 1], backend=backend)
    exp_data = exp.run(backend=backend).block_for_results()
    mean_success_probability = exp_data.analysis_results(
        "mean_success_probability", dataframe=True
    ).iloc[0]
    assert exp_data.job_status() == JobStatus.DONE
    assert exp_data.analysis_status() == AnalysisStatus.DONE
    assert mean_success_probability.value


def test_mirror_rb(backend):
    exp = MirrorRB(physical_qubits=[0, 1, 2], lengths=[2], backend=backend)
    exp_data = exp.run(backend=backend).block_for_results()
    assert exp_data.job_status() == JobStatus.DONE
    assert exp_data.analysis_status() == AnalysisStatus.DONE


def test_purity_rb(backend):
    exp = PurityRB(physical_qubits=[0, 1], lengths=[1], backend=backend)
    exp_data = exp.run(backend=backend).block_for_results()
    alpha = exp_data.analysis_results("alpha", dataframe=True).iloc[0]
    EPC = exp_data.analysis_results("EPC", dataframe=True).iloc[0]
    EPG_cz = exp_data.analysis_results("EPG_cz", dataframe=True).iloc[0]
    assert exp_data.job_status() == JobStatus.DONE
    assert exp_data.analysis_status() == AnalysisStatus.DONE
    assert alpha.value
    assert EPC.value
    assert EPG_cz.value
