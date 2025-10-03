# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from functools import lru_cache
from typing import Literal, Optional

import networkx as nx
import numpy as np
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray

from .mirror_circuits import mirror_trotter_pub_1d
from .get_optimal_path import get_optimal_path


def mirror_pub(
    num_theta: int,
    backend,
    num_qubits: Optional[int] = None,
    num_steps: int = None,
    target_num_2q_gates: int = None,
    num_magnetization: int = 1,
    repeat_magnetization: bool = False,
    repeat_theta: bool = False,
    path: tuple[int] = None,
    theta: float = np.pi / 4,
) -> EstimatorPub:
    if num_steps is None and target_num_2q_gates is None:
        raise ValueError("Must specify either num steps or target 2q gates")
    if num_steps is not None and target_num_2q_gates is not None:
        raise ValueError("Can only specify either num steps or target 2q gates")

    if path is None:
        coupling_map = tuple(tuple(edge) for edge in backend.coupling_map)
        maximal_path = get_longest_path(coupling_map)

        if num_qubits is None:
            num_qubits = len(maximal_path)
        path = tuple(maximal_path[:num_qubits])
    else:
        if len(path) < num_qubits:
            raise ValueError(
                f"Not enough qubits specified in path, {len(path)} < {num_qubits}"
            )
        path = path[:num_qubits]

    if target_num_2q_gates is not None:
        num_steps = int(np.round(0.5 * target_num_2q_gates / num_qubits))
    if num_steps is not None:
        target_num_2q_gates = int(np.round(2 * num_steps * num_qubits))

    if num_steps == 0:
        num_steps = 1
    if target_num_2q_gates == 0:
        target_num_2q_gates = int(np.round(2 * num_steps * num_qubits))

    if num_steps < 1 or target_num_2q_gates < 1:
        raise ValueError(
            f"Must have at least one step and 2q gate, got: {num_steps} steps and {target_num_2q_gates} 2q gates"
        )

    if num_theta == 1:
        theta = (theta,)
    else:
        theta = tuple(np.linspace(0, theta, num=num_theta))

    if num_magnetization == 1:
        magnetization = (1.0,)
    else:
        magnetization = tuple(np.linspace(0, 1, num=num_magnetization))

    if repeat_magnetization:
        magnetization = (1.0,) * num_magnetization
    if repeat_theta:
        theta = (1.0,) * num_theta

    pub = mirror_trotter_pub_1d(
        num_steps=num_steps,
        path=path,
        backend=backend,
        theta_values=theta,
        magnetization_values=magnetization,
    )

    if not (_gates := list(set(backend.basis_gates).intersection(["cx", "ecr", "cz"]))):
        raise ValueError("2q gate not recognized")
    else:
        gate_name = _gates[0]
    num_2q_gates = pub[0].count_ops().get(gate_name, 0)

    pub[0].metadata["circuit_depth"] = pub[0].depth(lambda instr: len(instr.qubits) > 1)
    pub[0].metadata["theta"] = theta
    pub[0].metadata["path"] = path
    pub[0].metadata["magnetization"] = magnetization
    pub[0].metadata["num_steps"] = num_steps
    pub[0].metadata["num_qubits"] = num_qubits
    pub[0].metadata["num_2q_gates"] = num_2q_gates
    pub[0].metadata["num_2q_gates_per_step_per_qubit"] = (
        num_2q_gates / num_steps
    ) / num_qubits

    pars_from_circ = tuple(pub[0].parameters)
    if str(pars_from_circ[0]) != "A":
        raise ValueError("Assumed parameter order violated")

    pub = list(pub)
    pub[1] = ObservablesArray(pub[1])
    pub[2] = BindingsArray({pars_from_circ: pub[2]})
    pub = tuple(pub)
    pub = EstimatorPub(*pub)

    pub.circuit.metadata["bindings_array_shape"] = pub.parameter_values.shape
    pub.circuit.metadata["observables_array_shape"] = pub.observables.shape

    return pub


@lru_cache()
def get_longest_path(coupling_map):
    graph = nx.Graph(list(coupling_map))
    maximal_path = max(nx.all_simple_paths(graph, 13, 113), key=len)
    return maximal_path


class MirrorPubOptions:
    num_qubits: Optional[int] = None
    target_num_2q_gates: Optional[int] = 1000
    num_steps: Optional[int] = None
    num_magnetization: int = 1
    num_theta: int = 1
    theta: float = np.pi / 4
    repeat_theta: bool = False
    repeat_magnetization: bool = False
    num_pubs: int = 1
    path: Optional[tuple[int, ...]] = None
    path_strategy: Literal[None, "vf2_optimal", "eplg_chain"] = None

    def get_pubs(self, backend) -> list[EstimatorPub]:
        pub = mirror_pub(
            backend=backend,
            num_qubits=self.num_qubits,
            target_num_2q_gates=self.target_num_2q_gates,
            num_steps=self.num_steps,
            num_theta=self.num_theta,
            num_magnetization=self.num_magnetization,
            repeat_magnetization=self.repeat_magnetization,
            repeat_theta=self.repeat_theta,
            path=self.get_path(backend),
            theta=self.theta,
        )

        return [pub] * self.num_pubs

    def get_path(self, backend):
        if self.path_strategy is None:
            if self.path:
                return self.path
            else:
                coupling_map = tuple(tuple(edge) for edge in backend.coupling_map)
                return get_longest_path(coupling_map)[: self.num_qubits]

        elif self.path_strategy == "eplg_chain":
            if self.num_qubits > 100:
                raise ValueError("ELPG chain only defined up to 100 qubits")
            eplg_chain = next(
                q_list["qubits"]
                for q_list in backend.properties().general_qlists
                if q_list["name"] == "lf_100"
            )
            return eplg_chain[: self.num_qubits]

        elif self.path_strategy == "vf2_optimal":
            weights_dict = {
                "t1": 1.0,
                "t2": 0,
                "readout_error": 1.0,
                "faulty": 0,
                "gate_err_2q": 1.0,
            }

            path = get_optimal_path(
                weights_dict=weights_dict,
                backend=backend,
                num_qubits=self.num_qubits,
                time_limit=1e2,
                seed=42,
                max_trials=1000,
            )

            return path
        else:
            raise ValueError(f"Unreconized path_strategy value {self.path_strategy}")
