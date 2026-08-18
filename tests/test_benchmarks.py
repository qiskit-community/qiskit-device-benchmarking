import pytest

from qiskit.providers.jobstatus import JobStatus
from qiskit_experiments.framework import AnalysisStatus
from qiskit_ibm_runtime.fake_provider import FakeFez, FakePerth

from qiskit.circuit import Barrier
from qiskit.transpiler import CouplingMap

import qiskit_device_benchmarking.utilities.graph_utils as gu
from qiskit_device_benchmarking.clops.clops_benchmark import (
    QUBIT_SET_SOURCE_REPORTED,
    QUBIT_SET_SOURCE_TOPOLOGY,
    QUBIT_SET_SOURCE_USER,
    chain_to_layers,
    clops_label,
    create_hardware_aware_circuit,
    get_reported_chain,
    pruned_coupling_map,
    select_qubit_set,
    validate_chain,
)
from qiskit_device_benchmarking.bench_code.bell import CHSHExperiment
from qiskit_device_benchmarking.mcps.mcps_benchmark import create_mcps_circuit, run_mcps_sampler

# from qiskit_device_benchmarking.bench_code.dynamic_circuits_rb import DynamicCircuitsRB
# from qiskit_device_benchmarking.bench_code.mcm_rb_experiment import McmRB
from qiskit_device_benchmarking.bench_code.mrb.mirror_qv import MirrorQuantumVolume
from qiskit_device_benchmarking.bench_code.mrb.mirror_rb_experiment import MirrorRB
from qiskit_device_benchmarking.bench_code.prb.pur_rb import PurityRB


@pytest.fixture
def backend():
    return FakeFez()


def _two_q_pairs_per_layer(qc):
    """Split a circuit on full width barriers and return the set of 2Q qubit
    pairs in each resulting slice, dropping slices with no 2Q gates."""
    slices = []
    current = set()
    for inst in qc.data:
        if isinstance(inst.operation, Barrier):
            if current:
                slices.append(current)
            current = set()
            continue
        if len(inst.qubits) == 2:
            current.add(tuple(sorted(qc.find_bit(q).index for q in inst.qubits)))
    if current:
        slices.append(current)
    return slices


def test_clops_hardware_aware_circuit(backend):
    for parameterized in (False, True):
        qc, params, metadata = create_hardware_aware_circuit(
            width=100,
            layers=100,
            backend=backend,
            parameterized=parameterized,
            return_metadata=True,
        )

        # the qubit set is a 100q chain, though the register spans the device
        assert len(metadata["qubits"]) == 100
        assert qc.num_clbits == 100
        assert metadata["num_sublayers"] == 2

        # append_1q_layer always returns its ParameterVectors; `parameterized`
        # controls whether the circuit actually binds them
        assert params
        assert bool(qc.parameters) is parameterized


def test_clops_chain_decomposition_matches_lf(backend):
    """The decomposition must match layer_fidelity_utils.run_lf_chain exactly."""
    coupling_map, faulty_qubits = pruned_coupling_map(backend)
    cm = CouplingMap(list(coupling_map))
    chain, _ = select_qubit_set(backend, 20)

    layer1, layer2 = chain_to_layers(chain, cm)

    all_pairs = [tuple(pair) for pair in gu.path_to_edges([list(chain)], cm)[0]]
    assert layer1 == all_pairs[0::2]
    assert layer2 == all_pairs[1::2]

    # a 20q chain has 19 bonds, split 10 / 9
    assert len(layer1) == 10
    assert len(layer2) == 9

    # each sublayer must be qubit disjoint
    for layer in (layer1, layer2):
        flat = [q for edge in layer for q in edge]
        assert len(flat) == len(set(flat))


def test_clops_chain_decomposition_rejects_bad_chain(backend):
    coupling_map, _ = pruned_coupling_map(backend)
    cm = CouplingMap(list(coupling_map))

    with pytest.raises(ValueError):
        chain_to_layers([0, 1, 155], cm)

    with pytest.raises(ValueError, match="at least 3 qubits"):
        chain_to_layers([0, 1], cm)


def test_clops_layers_cycle_two_sublayers(backend):
    """Every layer must reuse the fixed sublayers, alternating L1, L2, L1, L2."""
    qc, _, metadata = create_hardware_aware_circuit(
        width=20,
        layers=6,
        backend=backend,
        parameterized=False,
        return_metadata=True,
    )

    layer1, layer2 = [
        {tuple(sorted(edge)) for edge in sl} for sl in metadata["two_qubit_layers"]
    ]
    slices = _two_q_pairs_per_layer(qc)

    assert len(slices) == 6
    assert slices[0] == slices[2] == slices[4] == layer1
    assert slices[1] == slices[3] == slices[5] == layer2

    # 6 layers over a 20q chain: three L1 (10 gates) and three L2 (9 gates)
    n_2q = sum(
        1
        for inst in qc.data
        if len(inst.qubits) == 2 and not isinstance(inst.operation, Barrier)
    )
    assert n_2q == 3 * 10 + 3 * 9


def test_clops_distinct_parameters_per_layer(backend):
    """Structure repeats with period 2, but every depth position has its own
    parameters."""
    _, params, _ = create_hardware_aware_circuit(
        width=20,
        layers=6,
        backend=backend,
        parameterized=True,
        return_metadata=True,
    )

    # rzsx basis yields 3 ParameterVectors per layer
    assert len(params) == 6 * 3
    prefixes = {pv.name.rsplit("_", 1)[0] for pv in params}
    assert prefixes == {"L%d" % d for d in range(6)}

    # no parameter is shared between layers
    all_params = [p for pv in params for p in pv]
    assert len(set(all_params)) == 6 * 3 * 20


def test_clops_idle_qubits_per_sublayer(backend):
    """L1 covers the whole chain, L2 leaves both endpoints idle. This is why the
    twirled path keeps its explicit 1Q layer: gate twirling only covers qubits
    active in a box, so those endpoints would otherwise get no 1Q gate."""
    _, _, metadata = create_hardware_aware_circuit(
        width=100,
        layers=100,
        backend=backend,
        parameterized=False,
        return_metadata=True,
    )

    chain = set(metadata["qubits"])
    layer1, layer2 = metadata["two_qubit_layers"]

    assert len(layer1) == 50
    assert len(layer2) == 49

    # even bonds cover every qubit; odd bonds leave the two chain ends idle
    assert chain - {q for edge in layer1 for q in edge} == set()
    assert chain - {q for edge in layer2 for q in edge} == {
        metadata["qubits"][0],
        metadata["qubits"][-1],
    }


def test_clops_qubit_set_topology_fallback(backend, monkeypatch):
    """With no reported chain, selection falls back to the topology heuristic."""
    monkeypatch.setattr(
        "qiskit_device_benchmarking.clops.clops_benchmark.get_reported_chain",
        lambda backend, width: None,
    )

    chain, source = select_qubit_set(backend, 100)

    assert len(chain) == 100
    assert source == QUBIT_SET_SOURCE_TOPOLOGY

    coupling_map, faulty_qubits = pruned_coupling_map(backend)
    validate_chain(chain, coupling_map, faulty_qubits)


def test_clops_reported_chain_takes_precedence(backend):
    """FakeFez publishes lf_4 .. lf_100, so the reported chain is used by
    default and CLOPS lands on the same qubits as the EPLG measurement."""
    reported = get_reported_chain(backend, 100)
    assert reported is not None and len(reported) == 100

    chain, source = select_qubit_set(backend, 100)

    assert chain == [int(q) for q in reported]
    assert source == QUBIT_SET_SOURCE_REPORTED

    # and the reported chain is a valid path, so it decomposes cleanly
    coupling_map, faulty_qubits = pruned_coupling_map(backend)
    validate_chain(chain, coupling_map, faulty_qubits)


def test_clops_reported_chain_too_short_warns_and_falls_back(backend, monkeypatch):
    coupling_map, _ = pruned_coupling_map(backend)
    short = gu.longest_path_of_length(coupling_map, 50)

    monkeypatch.setattr(
        "qiskit_device_benchmarking.clops.clops_benchmark.get_reported_chain",
        lambda backend, width: short,
    )

    with pytest.warns(UserWarning, match="shorter than the requested width"):
        chain, source = select_qubit_set(backend, 100)

    assert len(chain) == 100
    assert source == QUBIT_SET_SOURCE_TOPOLOGY


def test_clops_explicit_qubits(backend):
    coupling_map, faulty_qubits = pruned_coupling_map(backend)
    explicit = gu.longest_path_of_length(coupling_map, 12)

    chain, source = select_qubit_set(backend, 12, qubits=explicit)

    assert chain == [int(q) for q in explicit]
    assert source == QUBIT_SET_SOURCE_USER

    # a chain whose length disagrees with the requested width is an error
    with pytest.raises(ValueError, match="but width is"):
        select_qubit_set(backend, 100, qubits=explicit)


def test_clops_validate_chain(backend):
    coupling_map, faulty_qubits = pruned_coupling_map(backend)

    with pytest.raises(ValueError, match="repeated qubits"):
        validate_chain([0, 1, 1], coupling_map, faulty_qubits)

    with pytest.raises(ValueError, match="Path not found"):
        validate_chain([0, 1, 155], coupling_map, faulty_qubits)


def test_clops_width_too_large(backend):
    with pytest.raises(ValueError, match="Insufficient connected qubits"):
        select_qubit_set(backend, 500)


@pytest.mark.parametrize(
    "width,layers,shots,expected",
    [
        (100, 100, 100, "CLOPS"),
        (100, 100, 50, "CLOPS_h(100, 100, 50)"),
        (50, 100, 100, "CLOPS_h(50, 100, 100)"),
    ],
)
def test_clops_label(width, layers, shots, expected):
    assert clops_label(width, layers, shots) == expected


@pytest.mark.parametrize("layer_order", ["2q_first", "1q_first"])
def test_clops_layer_orders(backend, layer_order):
    """Both orderings build and run the same number of 2Q gates, so the A/B
    comparison differs only in barrier placement."""
    qc, _, metadata = create_hardware_aware_circuit(
        width=20,
        layers=4,
        backend=backend,
        parameterized=False,
        layer_order=layer_order,
        return_metadata=True,
    )

    assert metadata["layer_order"] == layer_order

    n_2q = sum(
        1
        for inst in qc.data
        if len(inst.qubits) == 2 and not isinstance(inst.operation, Barrier)
    )
    assert n_2q == 2 * 10 + 2 * 9


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


def test_mcps_circuit():
    """Verify the MCPS circuit has H on every qubit followed by measurements."""
    small_backend = FakePerth()
    qc = create_mcps_circuit(small_backend)
    gate_ops = [
        inst.operation.name
        for inst in qc.data
        if inst.operation.name not in ("measure", "barrier")
    ]
    assert qc.num_qubits == small_backend.num_qubits
    assert set(gate_ops) <= {"h", "rz", "sx", "x"}  # H or its hardware decomposition
    assert len(gate_ops) >= small_backend.num_qubits  # at least one gate per qubit


def test_mcps_sampler():
    """Run a small MCPS job via the Sampler path and verify it completes."""
    small_backend = FakePerth()
    job = run_mcps_sampler(small_backend, num_circuits=5, shots=10)
    result = job.result()
    assert len(result) == 5
