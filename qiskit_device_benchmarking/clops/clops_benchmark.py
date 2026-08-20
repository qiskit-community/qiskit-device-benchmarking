# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals


import copy
import warnings
from functools import lru_cache
from typing import Callable, List, Optional, Sequence

import numpy as np
from qiskit import transpile
from qiskit.circuit import Barrier, Delay, ParameterVector, QuantumCircuit, Qubit
from qiskit.circuit.library import IGate, RZGate, SXGate, UGate
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import SamplerOptions
from qiskit_ibm_runtime.options import TwirlingOptions
from qiskit_ibm_runtime.ibm_backend import Backend
from qiskit_ibm_runtime.execution_span import ExecutionSpan, ExecutionSpans

import qiskit_device_benchmarking.utilities.graph_utils as gu

"""
CLOPS benchmark

This benchmark measures Circuit Layer Operations Per Seconds (CLOPS) of
parameterized utility scale  hardware efficient circuits.  CLOPS measures 
the steady state throughput of a large quantity of  these parameterized 
circuits that are  of width 100 qubits with 100 layers of gates. 
Each layer consists of two qubit gates across as many qubits
as possible that can be done in parallel, followed by a single qubit
gate(s) on every qubit to allow any arbitrary rotation, with those 
rotations being parameterized.  Parameters are applied to the circuit 
to generate a large number of  instantiated circuits to be executed on 
the quantum computer. It is up to the vendor on how to optimally execute 
these circuits for maximal throughput.  As such, the benchmark code 
provides several ways to measure CLOPS depending on the capability of 
the quantum computer. 

The "twirling" method uses the native parameterization of the Sampler
primitive to parameterize the circuit, and optimal batching of the 
circuits is assumed to be done by the Sampler, freeing the user
from having to optimize the batch size. The only requirement is
that the total number of circuits executed needs to be chosen to
get the system into a steady state to measure CLOPS.
The "parameterized" method is similar, but instead sends an already
parameterized circuit to the Sampler primitive, along with enough 
parameters to execute the specified number of circuits. Batching 
again is handled by the Sampler. This method requires larger bandwidth
to send in all of the necessary parameters.

The "instantiated" method is for systems that cannot natively 
handle parameterized circuits. In this case the circuit parameters
are bound locally and then sent to the quantum computer for execution.
This method requires the user to specify the desired size of each
batch of circuits (so that they can be sent together the quantum computer) 
as well as the number of local parallel pipelines to bind parameters and
create payloads in parallel. The user will need to tune both of these
parameters to try and optimize performance of the system. This will 
tend to be much slower than on systems that natively support parameterized
circuits

Qubit set selection follows the CLOPS_h benchmark execution protocol. When the
backend reports a layer fidelity chain (the best 100 qubit chain found during
EPLG characterization), that chain is used, so the speed measurement is taken on
the same qubits as the companion layer fidelity measurement and the two can be
interpreted jointly. When no such chain is reported the qubit set is selected by
a topology only heuristic, and this is recorded in the reported metadata.

The circuit is built from the canonical layer decomposition of that chain: two
qubit disjoint sublayers, L1 on the even bonds and L2 on the odd bonds, cycled
L1, L2, L1, L2, ... over the requested depth. This is the same decomposition used
by the layer fidelity experiment. The structure of every layer is fixed; only the
single qubit gate parameters vary between depth positions and between executions.

For further information see the clops_benchmark class
"""


"""
Functions for working with circuits of parameterized 1-qubit gates.
"""


def append_1q_layer(
    circuit: QuantumCircuit,
    qubits: Optional[Sequence[Qubit]] = None,
    basis: str = "rzsx",
    parameterized: bool = False,
    parameter_prefix: str = "θ",
) -> List[ParameterVector]:
    """Append a layer of 1-qubit gates on specified qubits with optional parameters

    Args:
        circuit: The circuit to append the layer to
        qubits: Optional, the qubits to add parameterized gates to.
                If None all qubits will be included.
        basis: The parameter basis for the parameterized gates.
        parameterized: If the gates are to be parameterized or fixed
        parameter_prefix: The prefix for the parameter vectors.

    Returns:
        The ordered list of ParameterVectors for the appended layer.
    """
    if qubits is None:
        qubits = circuit.qubits
    return _layer_basis_function(basis)(
        circuit, qubits, parameterized, parameter_prefix
    )


def _append_1q_layer_u(
    circuit, qubits: Sequence[Qubit], parameterized=True, param_prefix="θ"
) -> List[ParameterVector]:
    """Append a layer of parameterized 1-qubit gates on specified qubits."""
    size = len(qubits)
    pars0 = ParameterVector(f"{param_prefix}_0", size)
    pars1 = ParameterVector(f"{param_prefix}_1", size)
    pars2 = ParameterVector(f"{param_prefix}_2", size)

    for i, q in enumerate(qubits):
        if parameterized:
            circuit._append(UGate(pars0[i], pars1[i], pars2[i]), [q], [])
        else:
            circuit._append(UGate(1.0, -3.14 / 2, 3.14 / 2), [q], [])

    return pars0, pars1, pars2


def _append_1q_layer_rzsx(
    circuit, qubits: List[Qubit], parameterized, param_prefix="θ"
) -> List[ParameterVector]:
    """Append a layer of parameterized 1-qubit gates on specified qubits."""
    size = len(qubits)
    pars0 = ParameterVector(f"{param_prefix}_0", size)
    pars1 = ParameterVector(f"{param_prefix}_1", size)
    pars2 = ParameterVector(f"{param_prefix}_2", size)

    for i, q in enumerate(qubits):
        if parameterized:
            circuit._append(RZGate(pars0[i]), [q], [])
            circuit._append(SXGate(), [q], [])
            circuit._append(RZGate(pars1[i]), [q], [])
            circuit._append(SXGate(), [q], [])
            circuit._append(RZGate(pars2[i]), [q], [])
        else:
            circuit._append(SXGate(), [q], [])

    return pars0, pars1, pars2


def _append_1q_layer_pauli(
    circuit: QuantumCircuit, qubits: List[Qubit], parameterized, param_prefix="θ"
) -> List[ParameterVector]:
    """Append a layer of parameterized 1-qubit gates on specified qubits."""
    from pec_runtime.circuit.pauli_gate import PauliGate

    size = len(qubits)
    pars_z = ParameterVector(f"{param_prefix}_0", size)
    pars_x = ParameterVector(f"{param_prefix}_1", size)

    # As a work around for terra bug 8692 we apply 1-qubit PauliGates only
    # https://github.com/Qiskit/qiskit-terra/issues/8692
    # TODO: support non-parameterized if needed
    for qubit, pz, px in zip(qubits, pars_z, pars_x):
        circuit._append(PauliGate([pz], [px]), [qubit], [])
    return pars_z, pars_x


@lru_cache()
def _layer_basis_function(basis: str) -> Callable:
    """Return function for appending a parameterized layer"""
    _funcs = {
        "rzsx": _append_1q_layer_rzsx,
        "clifford": _append_1q_layer_rzsx,
        "u": _append_1q_layer_u,
        "pauli": _append_1q_layer_pauli,
    }
    if basis not in _funcs:
        raise ValueError(f"Invalid 1-qubit parameter basis {basis}")
    return _funcs[basis]


@lru_cache()
def _is_identity(cls) -> bool:
    """Return True if a gate class can be treated as identity."""
    if issubclass(cls, (IGate, Delay, Barrier)):
        return True
    return False


QUBIT_SET_SOURCE_USER = "user"
QUBIT_SET_SOURCE_REPORTED = "eplg_reported"
QUBIT_SET_SOURCE_TOPOLOGY = "auto:topology"

# Sources that count as an external specification of the qubit set. A CLOPS_h
# result must report whether the qubit set was externally specified (e.g. taken
# from an EPLG measurement) or automatically selected.
_EXTERNALLY_SPECIFIED = frozenset({QUBIT_SET_SOURCE_USER, QUBIT_SET_SOURCE_REPORTED})


def two_qubit_gate(backend):
    """Return the two qubit basis gate for a backend

    Args:
        backend: the backend to inspect

    Returns:
        one of "ecr", "cz" or "cx"
    """

    basis_gates = backend.configuration().basis_gates
    for gate in ("ecr", "cz", "cx"):
        if gate in basis_gates:
            return gate

    raise ValueError("No supported two qubit gate in basis gates %s" % basis_gates)


def pruned_coupling_map(backend):
    """Return the backend coupling map with faulty gates and faulty qubits removed

    Note this also removes edges touching a faulty qubit, not just faulty gates,
    since the chain search runs directly on the returned map.

    Args:
        backend: the backend to inspect

    Returns:
        (coupling_map, faulty_qubits)
    """

    faulty_qubits = list(backend.properties().faulty_qubits())
    faulty_gates = backend.properties().faulty_gates()
    faulty_edges = [tuple(gate.qubits) for gate in faulty_gates if len(gate.qubits) > 1]

    coupling_map = copy.deepcopy(backend.coupling_map)

    for edge in faulty_edges:
        if tuple(edge) in coupling_map:
            coupling_map.graph.remove_edge(edge[0], edge[1])

    for edge in list(coupling_map):
        if edge[0] in faulty_qubits or edge[1] in faulty_qubits:
            if tuple(edge) in coupling_map:
                coupling_map.graph.remove_edge(edge[0], edge[1])

    return coupling_map, faulty_qubits


def get_reported_chain(backend, width):
    """Return the layer fidelity chain reported by the backend, or None

    Wraps layer_fidelity_utils.get_lf_chain, which returns None when the named
    qubit list is absent, which covers backends reporting an empty list. This
    wrapper additionally guards backends whose properties object has no
    general_qlists attribute at all, or is missing properties entirely.

    Args:
        backend: the backend to inspect
        width: the chain length to look for

    Returns:
        list of qubits, or None if the backend reports no such chain
    """

    try:
        import qiskit_device_benchmarking.utilities.layer_fidelity_utils as lfu

        return lfu.get_lf_chain(backend, width)
    except (AttributeError, TypeError, KeyError):
        return None


def validate_chain(chain, coupling_map, faulty_qubits=None):
    """Check that a chain is a simple path of good qubits in the coupling map

    Args:
        chain: list of qubits
        coupling_map: CouplingMap to validate against
        faulty_qubits: list of faulty qubits

    Raises:
        ValueError: if the chain repeats a qubit, contains a faulty qubit, or is
            not a connected path in the coupling map
    """

    if len(set(chain)) != len(chain):
        raise ValueError("Qubit chain contains repeated qubits: %s" % chain)

    bad_qubits = [q for q in chain if q in set(faulty_qubits or ())]
    if bad_qubits:
        raise ValueError("Qubit chain contains faulty qubits %s" % bad_qubits)

    # raises ValueError if consecutive qubits are not an edge
    gu.path_to_edges([list(chain)], coupling_map)


def select_qubit_set(
    backend, width, qubits=None, coupling_map=None, faulty_qubits=None
):
    """Select the qubit set to benchmark and record where it came from

    Implements the qubit set selection step of the CLOPS_h protocol. In
    precedence order:

    1. An explicit chain passed by the caller.
    2. The layer fidelity chain reported by the backend, so that CLOPS is
       measured on the same qubits as the companion EPLG measurement. A longer
       reported chain is truncated to `width`; a reported chain shorter than
       `width` cannot anchor a width `width` result, so it is skipped with a
       warning.
    3. A topology only heuristic, which must be declared when reporting.

    Args:
        backend: the backend to benchmark
        width: number of qubits wanted
        qubits: optional explicit chain to use
        coupling_map: optional pruned coupling map (computed if not given)
        faulty_qubits: optional list of faulty qubits (computed if not given)

    Returns:
        (qubit_set, source) where source is one of "user", "eplg_reported" or
        "auto:topology"
    """

    if coupling_map is None or faulty_qubits is None:
        coupling_map, faulty_qubits = pruned_coupling_map(backend)

    # 1. explicit chain from the caller
    if qubits is not None:
        chain = [int(q) for q in qubits]
        if len(chain) != width:
            raise ValueError(
                "Explicit qubits has length %d but width is %d. Pass a chain of "
                "the requested width, or leave width at its default."
                % (len(chain), width)
            )
        return chain, QUBIT_SET_SOURCE_USER

    # 2. the chain reported by the backend from EPLG characterization
    reported = get_reported_chain(backend, width)
    if reported is not None:
        if len(reported) >= width:
            return [int(q) for q in reported[:width]], QUBIT_SET_SOURCE_REPORTED

        warnings.warn(
            "Backend reports a layer fidelity chain of %d qubits, which is "
            "shorter than the requested width %d, so it cannot anchor this "
            "result. Falling back to automatic selection; report this qubit set "
            "as automatically selected." % (len(reported), width),
            stacklevel=2,
        )

    # 3. topology only fallback
    chain = gu.longest_path_of_length(coupling_map, width, faulty_qubits)
    if len(chain) < width:
        raise ValueError(
            "Insufficient connected qubits to create set of %d qubits, longest "
            "chain found was %d" % (width, len(chain))
        )

    return [int(q) for q in chain], QUBIT_SET_SOURCE_TOPOLOGY


def chain_to_layers(chain, coupling_map):
    """Return the canonical two sublayer decomposition of a 1D chain

    Returns L1 on the even bonds of the chain and L2 on the odd bonds, matching
    the decomposition used by the layer fidelity experiment (see
    layer_fidelity_utils.run_lf_chain), so a CLOPS result and an EPLG result on
    the same chain describe the same layers.

    Args:
        chain: list of qubits forming a connected path
        coupling_map: CouplingMap, used to orient each edge as the device
            declares it

    Returns:
        list of two lists of qubit pairs

    Raises:
        ValueError: if the chain is too short, is not a path in the coupling map,
            or yields a sublayer that is not qubit disjoint
    """

    chain = [int(q) for q in chain]
    if len(chain) < 3:
        raise ValueError("chain must contain at least 3 qubits to decompose")

    all_pairs = gu.path_to_edges([list(chain)], coupling_map)[0]
    all_pairs = [tuple(pair) for pair in all_pairs]
    layers = [all_pairs[0::2], all_pairs[1::2]]

    for idx, layer in enumerate(layers):
        used_qubits = set()
        for qpair in layer:
            if tuple(qpair) not in coupling_map and list(qpair) not in coupling_map:
                raise ValueError("Gate on %s does not exist" % (qpair,))

            if used_qubits & set(qpair):
                raise ValueError(
                    "Sublayer L%d is not qubit disjoint at gate %s" % (idx + 1, qpair)
                )
            used_qubits.update(qpair)

    return layers


def append_2q_layer(qc, edges, two_q_gate):
    """
    Add one physical layer of two qubit gates on the given qubit disjoint edges.
    """
    gate_funcs = {"ecr": qc.ecr, "cz": qc.cz, "cx": qc.cx}
    gate_func = gate_funcs[two_q_gate]

    for edge in edges:
        gate_func(*edge)
    return


def create_hardware_aware_circuit(
    width: int,
    layers: int,
    backend,
    parameterized=True,
    rng=None,
    qubits: Optional[Sequence[int]] = None,
    layer_order: str = "2q_first",
    return_metadata: bool = False,
):
    """
    Creates a circuit with a structure of alternating 1Q and 2Q gate layers.

    The qubit set is a connected 1D chain, taken from the layer fidelity chain
    reported by the backend when one is available (see select_qubit_set). Its
    canonical two sublayer decomposition (even bonds, odd bonds) is cycled over
    the requested depth, so every layer has a fixed structure matching the
    companion layer fidelity measurement. 1Q gate layers are parameterized if
    `parameterized` is set to True, otherwise returns fixed circuit, and each
    depth position gets its own parameters.

    Args:
        width: number of qubits in the circuit
        layers: number of physical layers
        backend: the backend to build for
        parameterized: whether the 1Q layers carry parameters
        rng: unused, retained for backwards compatibility
        qubits: optional explicit qubit chain to use, highest precedence
        layer_order: "2q_first" places the barrier between the 2Q and 1Q gates,
            which is the twirling box the Sampler's gate twirling expects.
            "1q_first" instead orders each layer as 1Q gates, 2Q gates, barrier,
            terminating every physical layer with the barrier.
        return_metadata: if True, also return a dict describing the qubit set,
            its source and the layer decomposition, for result reporting

    Returns:
        (circuit, param_list) or (circuit, param_list, metadata)
    """
    if width < 3:
        raise ValueError("'width' must be at least 3 to form a chain")
    if layers < 1:
        raise ValueError("'layers' must be at least 1")
    if layer_order not in ("2q_first", "1q_first"):
        raise ValueError("'layer_order' " + layer_order + " invalid")

    coupling_map, faulty_qubits = pruned_coupling_map(backend)

    qubit_map, qubit_source = select_qubit_set(
        backend,
        width,
        qubits=qubits,
        coupling_map=coupling_map,
        faulty_qubits=faulty_qubits,
    )
    validate_chain(qubit_map, coupling_map, faulty_qubits)

    # the canonical decomposition needs edges oriented as the device declares
    # them, which requires a CouplingMap rather than a bare list of edges
    sublayers = chain_to_layers(qubit_map, CouplingMap(list(coupling_map)))
    two_q_gate = two_qubit_gate(backend)

    qc = QuantumCircuit(max(qubit_map) + 1, width)

    qubits_obj = [qc.qubits[i] for i in qubit_map]
    param_list = []
    for d in range(layers):
        # cycle the fixed sublayers: L1, L2, L1, L2, ...
        edges = sublayers[d % len(sublayers)]

        if layer_order == "1q_first":
            # add single qubit gate layer with optional parameters
            param_list += append_1q_layer(
                qc,
                qubits=qubits_obj,
                parameterized=parameterized,
                parameter_prefix="L" + str(d),
            )

            append_2q_layer(qc, edges, two_q_gate)

            # barrier terminates the physical layer
            qc.barrier(qubits_obj)
        else:
            append_2q_layer(qc, edges, two_q_gate)

            # add barrier to form "twirling box" to inform primitve where
            # layers are for twirled gates
            qc.barrier(qubits_obj)

            # add single qubit gate layer with optional parameters
            param_list += append_1q_layer(
                qc,
                qubits=qubits_obj,
                parameterized=parameterized,
                parameter_prefix="L" + str(d),
            )

    qc.barrier(qubits_obj)
    transpiled_circ = transpile(
        qc, backend, translation_method="translator", layout_method="trivial"
    )

    for idx in range(width):
        transpiled_circ.measure(qubit_map[idx], idx)

    if return_metadata:
        metadata = {
            "qubits": list(qubit_map),
            "qubit_set_source": qubit_source,
            "externally_specified": qubit_source in _EXTERNALLY_SPECIFIED,
            "two_qubit_layers": [[list(edge) for edge in sl] for sl in sublayers],
            "num_sublayers": len(sublayers),
            "two_qubit_gate": two_q_gate,
            "layer_order": layer_order,
        }
        return transpiled_circ, param_list, metadata

    return transpiled_circ, param_list


def create_payload(backend, qc, rng, max_experiments):
    """
    Creates a payload of instantiated parameterized circuits that maximizes
    throughput
    """
    if max_experiments is None:
        max_experiments = backend.configuration().max_experiments
    job = []

    for idx in range(max_experiments):
        val = {}
        for param in qc.parameters:
            val[param] = rng.uniform(0, np.pi * 2)
        job.append(qc.assign_parameters(val))
    return job, max_experiments


def run_twirled(
    backend: Backend,
    width: int,
    layers: int,
    shots: int,
    rep_delay: float,
    num_circuits: int,
    execution_path: str,
    circuit_kwargs: Optional[dict] = None,
):
    """Run CLOPS letting the Sampler's gate twirling parameterize the circuit.

    The submitted circuit is unparameterized, so its 1Q layers are a single fixed
    SX per qubit. The Sampler supplies the parameterization: for each twirling box
    it inserts an rz-sx-rz-sx-rz frame ahead of the 2Q gates, which is the fully
    twirled single qubit layer the CLOPS_h protocol specifies.

    The fixed SX layer is therefore additional to what the protocol requires, and
    is deliberately kept. Gate twirling only covers the qubits taking part in a
    box, so a qubit idling in a given sublayer receives no twirl frame, while the
    protocol asks for a single qubit gate on every qubit in the set including idle
    ones. On a 100 qubit chain the 50 even bonds of L1 cover every qubit, but the
    49 odd bonds of L2 leave both chain endpoints idle. Keeping the SX layer means
    those qubits are never left bare, at the cost of slightly slowing the circuit.
    """
    (transpiled_circ, _, metadata) = create_hardware_aware_circuit(
        width=width,
        layers=layers,
        backend=backend,
        parameterized=False,
        return_metadata=True,
        **(circuit_kwargs or {}),
    )
    twirling_opts = TwirlingOptions(
        num_randomizations=num_circuits,
        shots_per_randomization=shots,
        enable_gates=True,
    )
    experimental_opts = {"execution": {"fast_parametric_update": True}}
    options = SamplerOptions(twirling=twirling_opts, experimental=experimental_opts)
    if execution_path:
        options.experimental["execution_path"] = execution_path

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session, options=options)
        job = sampler.run([transpiled_circ], shots=shots * num_circuits)

    metadata["twirling"] = {
        "enable_gates": True,
        "num_randomizations": num_circuits,
        "shots_per_randomization": shots,
    }
    metadata["parameterization"] = "sampler_gate_twirling"

    return job, metadata


def run_parameterized(
    backend: Backend,
    width: int,
    layers: int,
    shots: int,
    rep_delay: float,
    num_circuits: int,
    execution_path: str,
    circuit_kwargs: Optional[dict] = None,
):
    (transpiled_circ, parameters, metadata) = create_hardware_aware_circuit(
        width=width,
        layers=layers,
        backend=backend,
        parameterized=True,
        return_metadata=True,
        **(circuit_kwargs or {}),
    )
    seed = 234987
    rng = np.random.default_rng(seed)

    param_values = [
        [
            rng.uniform(0, np.pi * 2)
            for idx in range(sum([len(param) for param in parameters]))
        ]
        for idx in range(num_circuits)
    ]

    experimental_opts = {"execution": {"fast_parametric_update": True}}
    options = SamplerOptions(experimental=experimental_opts)
    if execution_path:
        options.experimental["execution_path"] = execution_path

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session, options=options)
        job = sampler.run([(transpiled_circ, param_values, shots)])

    metadata["twirling"] = {"enable_gates": False}
    metadata["parameterization"] = "rzsx_3param_per_qubit_per_layer"

    return job, metadata


def clops_label(width, layers, shots):
    """Return the label a CLOPS result should be reported under

    An unqualified "CLOPS" denotes the canonical operating point of 100 qubits,
    100 layers and 100 shots. Any other operating point carries its parameters in
    the label so the operating point travels with the number.

    Args:
        width: number of qubits
        layers: number of physical layers
        shots: shots per circuit execution

    Returns:
        "CLOPS" or "CLOPS_h(N, D, S)"
    """

    if width == 100 and layers == 100 and shots == 100:
        return "CLOPS"

    return "CLOPS_h(%d, %d, %d)" % (width, layers, shots)


class clops_benchmark:
    def __init__(
        self,
        service: QiskitRuntimeService,
        backend_name: str,
        width: int = 100,
        layers: int = 100,
        shots: int = 100,
        rep_delay: float = None,
        num_circuits: int = 5000,
        circuit_type: str = "twirled",
        batch_size: int = None,
        pipelines: int = 1,
        execution_path: Optional[str] = None,
        qubits: Optional[Sequence[int]] = None,
        layer_order: str = "2q_first",
        lfoc_declared: Optional[bool] = None,
    ):
        """Run CLOPS benchmark through Sampler primitive

        Args:
            service: instantiated service to use to run the benchmark
            backend_name: The backend to run the benchmark
            width: Optional, width of the CLOPS circuit, default is 100 qubits wide
            layers: Optional, number of layers in the CLOPS circuit, default is 100
                    which yields a 5K circuit when combined with 100 qubits wide
            shots: Optional, number of shots per circuit, default is 100
            rep_delay: Optional, delay between circuits, default is set to system value
            num_circuits: Optional, number of circuits (parameter updates) run for the benchmark
                          default is 5000.  Adjust as necessary to get sufficient iterations.
                          For non-twirled benchmarking may need to be significantly reduced to
                          meet API input size limits
            circuit_type: Optional, determines how parameters are handled:
                        "twirled": default value, sends in a single unparameterized circuit and configures
                                   Sampler to run `num_circuits` twirls (parameterized) of the circuit
                        "parameterized": sends in a single parameterized circuit with `num_circuits` parameter sets
                        "instantiated": binds parameters locally and sends batches of instantiated circuits to be run
            batch_size: Optional, indicates how many circuits should be sent in per job to the backend. Only used
                        for `instantiated` circuit_type. Default is None
            pipelines: Optional, number of parallel processes used to instantiate parameters and submit jobs to
                    the backend. Only used for `instantiated` circuit_type.  Default is 1
            execution_path: Optional, A value to pass to the experimental "execution_path" option of the
                        Sampler
            qubits: Optional, an explicit qubit chain to benchmark. Takes precedence over
                        the layer fidelity chain reported by the backend. Must be a connected
                        path of length `width`
            layer_order: Optional, ordering within each physical layer. Default "2q_first"
                        keeps the barrier between the 2Q and 1Q gates, which is the twirling
                        box the Sampler's gate twirling expects. "1q_first" orders each layer
                        as 1Q gates, 2Q gates, barrier
            lfoc_declared: Optional, whether this run satisfies layer fidelity operating
                        conditions. CLOPS does not verify this internally, so it must be
                        declared when reporting a result. Left as None if not stated
        """

        # service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend(backend_name)
        if rep_delay is None:
            rep_delay = backend.configuration().default_rep_delay

        circuit_kwargs = {"qubits": qubits, "layer_order": layer_order}

        if circuit_type == "twirled":
            self.job, metadata = run_twirled(
                backend,
                width,
                layers,
                shots,
                rep_delay,
                num_circuits,
                execution_path,
                circuit_kwargs,
            )
            self.clops = self._clops_throughput_sampler
        elif circuit_type == "parameterized":
            self.job, metadata = run_parameterized(
                backend,
                width,
                layers,
                shots,
                rep_delay,
                num_circuits,
                execution_path,
                circuit_kwargs,
            )
            self.clops = self._clops_throughput_sampler
        elif circuit_type == "instantiated":
            raise ValueError("'circuit_type' instantiated not yet supported")
        else:
            raise ValueError("'circuit_type' " + circuit_type + " invalid")

        self.job_attributes = {
            "backend_name": backend_name,
            "width": width,
            "layers": layers,
            "shots": shots,
            "rep_delay": rep_delay,
            "num_circuits": num_circuits,
            "circuit_type": circuit_type,
            "batch_size": batch_size,
            "pipelines": pipelines,
            # CLOPS_h reporting requirements
            "qubits": metadata["qubits"],
            "qubit_set_source": metadata["qubit_set_source"],
            "externally_specified": metadata["externally_specified"],
            "two_qubit_layers": metadata["two_qubit_layers"],
            "num_sublayers": metadata["num_sublayers"],
            "two_qubit_gate": metadata["two_qubit_gate"],
            "layer_order": metadata["layer_order"],
            "twirling": metadata["twirling"],
            "parameterization": metadata["parameterization"],
            "execution_interface": "qiskit-ibm-runtime SamplerV2",
            "execution_path": execution_path,
            "canonical_operating_point": (
                width == 100 and layers == 100 and shots == 100
            ),
            "clops_label": clops_label(width, layers, shots),
            "lfoc_declared": lfoc_declared,
        }

    def _clops_throughput_sampler(self):
        """Measures the overall CLOPS throughput based off of intermediate
        job metadata returned from the sampler. This metadata indicates the
        start and end time for each sub-job executed on the qpu.  For larger
        jobs (large number of twirls or large number of circuits/parameters)
        jobs are split into chunks that efficiently run on the qpu. To calculate
        the steady state throughput we use the time from the end of the first
        sub-job to the end of the last sub-job, skipping the startup costs
        for the first job to get the pipeline full. The goal is to predict
        the expected performance for large scale error mitigated workloads
        without having to run huge number of circuits"""

        result = self.job.result()
        execution_spans: ExecutionSpans = result.metadata["execution"][
            "execution_spans"
        ]
        spans = execution_spans.sort()
        span: ExecutionSpan
        sum_size: int = 0

        for span in spans:
            sum_size += span.size

        end_time_last_sub_job = spans.stop
        end_time_first_sub_job = spans[0].stop

        clops = round(
            ((sum_size - spans[0].size) * self.job_attributes["layers"])
            / (end_time_last_sub_job - end_time_first_sub_job).total_seconds()
        )

        return clops

    def report(self):
        """Return the fields a CLOPS result must specify when reported

        Note this blocks until the job result is available, since it calls
        clops(). CLOPS does not verify layer fidelity operating conditions
        internally, so lfoc_declared must be supplied by the caller; a warning is
        issued if it was never stated.

        Returns:
            dict of the reportable fields
        """

        attrs = self.job_attributes

        if attrs["lfoc_declared"] is None:
            warnings.warn(
                "lfoc_declared was not set, so this result does not declare "
                "layer fidelity operating conditions compliance. Pass "
                "lfoc_declared=True or False to state it.",
                stacklevel=2,
            )

        if not attrs["externally_specified"]:
            warnings.warn(
                "The qubit set was automatically selected (%s) rather than taken "
                "from a layer fidelity measurement. Declare this when reporting."
                % attrs["qubit_set_source"],
                stacklevel=2,
            )

        return {
            "label": attrs["clops_label"],
            "clops": self.clops(),
            "units": "physical layers per second",
            "backend_name": attrs["backend_name"],
            "width": attrs["width"],
            "layers": attrs["layers"],
            "shots": attrs["shots"],
            "qubits": attrs["qubits"],
            "qubit_set_source": attrs["qubit_set_source"],
            "externally_specified": attrs["externally_specified"],
            "parameterization": attrs["parameterization"],
            "twirling": attrs["twirling"],
            "layer_order": attrs["layer_order"],
            "execution_interface": attrs["execution_interface"],
            "canonical_operating_point": attrs["canonical_operating_point"],
            "lfoc_declared": attrs["lfoc_declared"],
        }
