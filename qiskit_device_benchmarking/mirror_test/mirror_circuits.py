# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Trotter circuit generation"""

from collections import defaultdict
from typing import Sequence
from math import inf
import numpy as np
import networkx as nx
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import CXGate, CZGate, ECRGate
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager as generate_pm
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.quantum_info import PauliList
from qiskit_ibm_runtime import IBMBackend


def remove_qubit_couplings(
    couplings: Sequence[tuple[int, int]], qubits: Sequence[int] | None = None
) -> list[tuple[int, int]]:
    """Remove qubits from a coupling list.

    Args:
        couplings: A sequence of qubit couplings.
        qubits: Optional, the qubits to remove.

    Returns:
        The input couplings with the specified qubits removed.
    """
    if qubits is None:
        return couplings
    qubits = set(qubits)
    return [edge for edge in couplings if not qubits.intersection(edge)]


def coupling_qubits(
    *couplings: Sequence[tuple[int, int]], allowed_qubits: Sequence[int] | None = None
) -> list[int]:
    """Return a sorted list of all qubits involved in 1 or more couplings lists.

    Args:
        couplings: 1 or more coupling lists.
        allowed_qubits: Optional, the allowed qubits to include. If None all
            qubits are allowed.

    Returns:
        The intersection of all qubits in the couplings and the allowed qubits.
    """
    qubits = set()
    for edges in couplings:
        for edge in edges:
            qubits.update(edge)
    if allowed_qubits is not None:
        qubits = qubits.intersection(allowed_qubits)
    return list(qubits)


def chain_coupling_map(
    coupling_map: list[tuple[int, int]],
    path: list[int],
) -> list[list[tuple[int, int]]]:
    """Construct the sub-CouplingMap for a 1D path through a 2D coupling map.

    Args:
        coupling_map: The input coupling map that is connected along the specified path.
        path: The ordered list of nodes to constructed a path for.

    Returns:
        The sub set of edges in the coupling map that are on the specified path.
    """
    coupling_edges = nx.DiGraph(list(coupling_map)).edges()
    path_edges = []
    for pos in range(1, len(path)):
        node_a = path[pos - 1]
        node_b = path[pos]
        for edge in ((node_a, node_b), (node_b, node_a)):
            if edge in coupling_edges:
                path_edges.append(edge)
    return path_edges


def directed_coupling_map(backend: IBMBackend) -> CouplingMap:
    """Construct a single-directional coupling map of shortest gates.

    Args:
        backend: A backend to extract coupling map and gate durations from.

    Returns:
        The directed coupling map of the shortest gate for each coupling pair.
    """
    directional_coupling = {}
    target = backend.target
    durations = target.durations()
    for inst, qubits in target.instructions:
        if inst.num_qubits == 2 and qubits is not None:
            key = tuple(sorted(qubits))
            if key in directional_coupling:
                continue
            q0, q1 = key
            try:
                length1 = durations.get(inst, (q0, q1))
            except TranspilerError:
                length1 = inf
            try:
                length2 = durations.get(inst, (q1, q0))
            except TranspilerError:
                length2 = inf

            shortest_pair = [q0, q1] if length1 <= length2 else [q1, q0]
            directional_coupling[key] = shortest_pair
    return CouplingMap(sorted(directional_coupling.values()))


def construct_layer_couplings(
    backend: IBMBackend, path: Sequence[int] = None
) -> list[list[tuple[int, int]]]:
    """Separate a coupling map into disjoint 2-qubit gate layers.

    Args:
        backend: A backend to construct layer couplings for.
        path: Optional, the ordered list of nodes to constructed a 1D path for couplings.

    Returns:
        A list of disjoint layers of directed couplings for the input coupling map.
    """
    coupling_map = directed_coupling_map(backend)
    if path is not None:
        coupling_map = chain_coupling_map(coupling_map, path)

    # Convert coupling map to a networkx graph
    coupling_graph = nx.Graph(list(coupling_map))

    # Edge coloring is vertex coloring on the dual graph
    dual_graph = nx.line_graph(coupling_graph)
    edge_coloring = nx.greedy_color(dual_graph, interchange=True)

    # Sort layers
    layers = defaultdict(list)
    for edge, color in edge_coloring.items():
        if edge not in coupling_map:
            edge = tuple(reversed(edge))
        layers[color].append(edge)
    layers = [sorted(layers[i]) for i in sorted(layers.keys())]

    return layers


def entangling_layer(
    gate_2q: str,
    couplings: Sequence[tuple[int, int]],
    qubits: Sequence[int] | None = None,
) -> QuantumCircuit:
    """Generating a entangling layer for the specified couplings.

    This corresonds to a Trotter layer for a ZZ Ising term with angle Pi/2.

    Args:
        gate_2q: The 2-qubit basis gate for the layer, should be "cx", "cz", or "ecr".
        couplings: A sequence of qubit couplings to add CX gates to.
        qubits: Optional, the physical qubits for the layer. Any couplings involving
            qubits not in this list will be removed. If None the range up to the largest
            qubit in the couplings will be used.

    Returns:
        The QuantumCircuit for the entangling layer.
    """
    # Get qubits and convert to set to order
    if qubits is None:
        qubits = range(1 + max(coupling_qubits(*couplings)))
    qubits = set(qubits)

    # Mapping of physical qubit to virtual qubit
    qubit_mapping = {q: i for i, q in enumerate(qubits)}

    # Convert couplings to indices for virtual qubits
    indices = [
        [qubit_mapping[i] for i in edge]
        for edge in couplings
        if qubits.issuperset(edge)
    ]

    # Layer circuit on virtual qubits
    circuit = QuantumCircuit(len(qubits))

    # Get 2-qubit basis gate and pre and post rotation circuits
    gate2q = None
    pre = QuantumCircuit(2)
    post = QuantumCircuit(2)

    if gate_2q == "cx":
        gate2q = CXGate()
        # Pre-rotation
        pre.sdg(0)
        pre.z(1)
        pre.sx(1)
        pre.s(1)
        # Post-rotation
        post.sdg(1)
        post.sxdg(1)
        post.s(1)
    elif gate_2q == "ecr":
        gate2q = ECRGate()
        # Pre-rotation
        pre.z(0)
        pre.s(1)
        pre.sx(1)
        pre.s(1)
        # Post-rotation
        post.x(0)
        post.sdg(1)
        post.sxdg(1)
        post.s(1)
    elif gate_2q == "cz":
        gate2q = CZGate()
        # Identity pre-rotation
        # Post-rotation
        post.sdg([0, 1])
    else:
        raise ValueError(
            f"Invalid 2-qubit basis gate {gate_2q}, should be 'cx', 'cz', or 'ecr'"
        )

    # Add 1Q pre-rotations
    for inds in indices:
        circuit.compose(pre, qubits=inds, inplace=True)

    # Use barriers around 2-qubit basis gate to specify a layer for PEA noise learning
    circuit.barrier()
    for inds in indices:
        circuit.append(gate2q, (inds[0], inds[1]))
    circuit.barrier()

    # Add 1Q post-rotations after barrier
    for inds in indices:
        circuit.compose(post, qubits=inds, inplace=True)

    # Add physical qubits as metadata
    circuit.metadata["physical_qubits"] = tuple(qubits)

    return circuit


def trotter_circuit(
    theta: Parameter | float,
    layer_couplings: Sequence[Sequence[tuple[int, int]]],
    num_steps: int,
    gate_2q: str | None = "cx",
    backend: IBMBackend | None = None,
    qubits: Sequence[int] | None = None,
) -> QuantumCircuit:
    """Generate a Trotter circuit for the 2D Ising

    Args:
        theta: The angle parameter for X.
        layer_couplings: A list of couplings for each entangling layer.
        num_steps: the number of Trotter steps.
        gate_2q: The 2-qubit basis gate to use in entangling layers.
            Can be "cx", "cz", "ecr", or None if a backend is provided.
        backend: A backend to get the 2-qubit basis gate from, if provided
            will override the basis_gate field.
        qubits: Optional, the allowed physical qubits to truncate the
            couplings to. If None the range up to the largest
            qubit in the couplings will be used.

    Returns:
        The Trotter circuit.
    """
    if backend is not None:
        try:
            basis_gates = backend.configuration().basis_gates
        except AttributeError:
            basis_gates = backend.basis_gates
        for gate in ["cx", "cz", "ecr"]:
            if gate in basis_gates:
                gate_2q = gate
                break

    # If no qubits, get the largest qubit from all layers and
    # specify the range so the same one is used for all layers.
    if qubits is None:
        qubits = range(1 + max(coupling_qubits(*layer_couplings)))

    coup_q_list = coupling_qubits(*layer_couplings)

    # Generate the entangling layers
    layers = [
        entangling_layer(gate_2q, couplings, qubits=qubits)
        for couplings in layer_couplings
    ]

    # Construct the circuit for a single Trotter step
    num_qubits = len(qubits)
    trotter_step = QuantumCircuit(num_qubits)
    trotter_step.rx(theta, coup_q_list)
    for layer in layers:
        trotter_step.compose(layer, range(num_qubits), inplace=True)

    # Construct the circuit for the specified number of Trotter steps
    circuit = QuantumCircuit(num_qubits)
    for _ in range(num_steps):
        circuit.rx(theta, coup_q_list)
        for layer in layers:
            circuit.compose(layer, range(num_qubits), inplace=True)

    circuit.metadata["physical_qubits"] = tuple(qubits)
    return circuit


def mirror_trotter_circuit_1d(
    theta: Parameter | float,
    delta: Parameter | float,
    num_steps: int,
    path: Sequence[int],
    backend: IBMBackend,
) -> QuantumCircuit:
    """Generate a mirrored Trotter circuit for 1D Ising on specified path.

    Args:
        theta: The angle parameter for X in simulated Hamiltonian.
        delta: The angle parameter for Rx rotation before final measurement
        num_steps: the number of Trotter steps. The returned circuit will
            have a 2-qubit layer depth of ``4 * num_steps``.
        path: The ordered list of nodes to constructed a 1D path for couplings.
        backend: A backend to get the 2-qubit basis gate from, if provided
            will override the basis_gate field.

    Returns:
        The Trotter circuit.
    """
    layer_couplings = construct_layer_couplings(backend, path=path)
    circuit = trotter_circuit(theta, layer_couplings, num_steps, backend=backend)

    # Construct mirror circuit
    mirror_circuit = circuit.compose(circuit.inverse())
    mirror_circuit.metadata["physical_qubits"] = path

    # Add contrast rotation
    mirror_circuit.rx(delta, path)

    return mirror_circuit


def mirror_trotter_pub_1d(
    num_steps: int,
    path: Sequence[int],
    backend: IBMBackend,
    theta_values: Sequence[float] = (0,),
    magnetization_values: Sequence[float] = (1,),
):
    """Generate a mirrored Trotter circuit EstimatorPub for 1D Ising on specified path.

    Args:
        num_steps: the number of Trotter steps. The returned circuit will
            have a 2-qubit layer depth of ``4 * num_steps``.
        path: The ordered list of nodes to constructed a 1D path for couplings.
        backend: A backend to get the 2-qubit basis gate from, if provided
            will override the basis_gate field.
        theta_values: The angle parameter values for X in simulated Hamiltonian.
        magnetization_values: The ideal magnetization values for the circuit after
            mirroring. Implement by single-qubit rotations of each qubit to reduce
            contrast of final measurements.

    Returns:
        The mirrored Trotter circuit EstimatorPub.
    """
    # Shape is theta first, magnetization second, observable last
    shape = (len(theta_values), len(magnetization_values), 1)
    theta_bc = np.broadcast_to(np.array(theta_values).reshape((-1, 1, 1)), shape)
    delta_vals = 2 * np.arccos(np.sqrt((1 + np.array(magnetization_values)) / 2))
    delta_bc = np.broadcast_to(delta_vals.reshape(1, -1, 1), shape)
    param_vals = np.stack([theta_bc, delta_bc], axis=-1)

    # Important that theta < delta in circuit order
    theta = Parameter("A")
    delta = Parameter("C")
    circuit = mirror_trotter_circuit_1d(theta, delta, num_steps, path, backend=backend)
    pm = generate_pm(
        optimization_level=1, target=backend.target, layout_method="trivial"
    )
    circuit = pm.run(circuit)
    obs = magnetization_observables(path, circuit.num_qubits)
    pub = (circuit, obs, param_vals)
    return pub


def magnetization_observables(
    physical_qubits: Sequence[int], num_qubits: int | None = None
) -> PauliList:
    """Return the PauliList for magnetization measurement observables for ISA circuits."""
    max_qubit = max(physical_qubits)
    if num_qubits is None:
        num_qubits = 1 + max_qubit
    elif num_qubits <= max_qubit:
        raise ValueError(
            f"num_qubits must be >= {max_qubit} for specified physical qubits"
        )
    zs = np.zeros((len(physical_qubits), num_qubits), dtype=bool)
    for idx, qubit in enumerate(physical_qubits):
        zs[idx, qubit] = True
    xs = np.zeros_like(zs)
    return PauliList.from_symplectic(zs, xs)
