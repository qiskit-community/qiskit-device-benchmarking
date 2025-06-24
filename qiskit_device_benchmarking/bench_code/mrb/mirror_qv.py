# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum Volume Experiment class.
"""

import warnings
from typing import Union, Sequence, Optional, List
from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence


from qiskit.circuit import (
    QuantumCircuit,
    ClassicalRegister,
)

from qiskit.circuit.library import QuantumVolume as QuantumVolumeCircuit
from qiskit.circuit.library import XGate
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, Options
from .mirror_qv_analysis import MirrorQuantumVolumeAnalysis

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import random_pauli_list, random_unitary


from qiskit.transpiler import PassManager, InstructionDurations
from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit_ibm_runtime.transpiler.passes.scheduling import PadDynamicalDecoupling
from qiskit_experiments.exceptions import QiskitError


class MirrorQuantumVolume(BaseExperiment):
    """Mirror Quantum Volume Experiment class.

    # section: overview
        Quantum Volume (QV) is a single-number metric that can be measured using a concrete protocol
        on near-term quantum computers of modest size. The QV method quantifies the largest random
        circuit of equal width and depth that the computer successfully implements.
        Quantum computing systems with high-fidelity operations, high connectivity,
        large calibrated gate sets, and circuit rewriting toolchains are expected to
        have higher quantum volumes.

        The Quantum Volume is determined by the largest circuit depth :math:`d_{max}`,
        and equals to :math:`2^{d_{max}}`.
        See `Qiskit Textbook
        <https://qiskit.org/textbook/ch-quantum-hardware/measuring-quantum-volume.html>`_
        for an explanation on the QV protocol.

        In the QV experiment we generate :class:`~qiskit.circuit.library.QuantumVolume` circuits on
        :math:`d` qubits, which contain :math:`d` layers, where each layer consists of random 2-qubit
        unitary gates from :math:`SU(4)`, followed by a random permutation on the :math:`d` qubits.
        Then these circuits run on the quantum backend and on an ideal simulator (either
        :class:`~qiskit.providers.aer.AerSimulator` or :class:`~qiskit.quantum_info.Statevector`).

        A depth :math:`d` QV circuit is successful if it has 'mean heavy-output probability' > 2/3 with
        confidence level > 0.977 (corresponding to z_value = 2), and at least 100 trials have been ran.

        See :class:`MirrorQuantumVolumeAnalysis` documentation for additional
        information on QV experiment analysis.

    # section: analysis_ref
        :py:class:`MirrorQuantumVolumeAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1811.12926
        .. ref_arxiv:: 2 2008.08571

    """

    def __init__(
        self,
        qubits: Sequence[int],
        backend: Optional[Backend] = None,
        trials: Optional[int] = 100,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        pauli_randomize: Optional[bool] = True,
        pauli_randomize_barriers: Optional[bool] = False,
        left_and_right: Optional[bool] = False,
        he: Optional[bool] = False,
    ):
        """Initialize a quantum volume experiment.

        Args:
            qubits: list of physical qubits for the experiment.
            backend: Optional, the backend to run the experiment on.
            trials: The number of trials to run the quantum volume circuit.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
            pauli_randomize: If True, add random Paulis to the beginning and end of
                a mirrored QV circuit
            pauli_randomize_barriers: If True, add barriers between the Paulis from
                pauli_randomize and the SU(4) elements
            left_and_right: If True, construct mirrored QV circuits from the left
                and right halves QV circuits. Circuits constructed from the right
                half have their inverses prepended, rather than appended.
            he: If true use a hardware efficient circuit (TwoLocal)

        Raises:
            Warning: if user attempts to split_inverse a QV experiment with odd depth
        """
        super().__init__(
            qubits, analysis=MirrorQuantumVolumeAnalysis(), backend=backend
        )

        # Set configurable options
        self.set_experiment_options(trials=trials, seed=seed)

        self.split_inverse = True
        # always set pauli_randomize to False if split_inverse is False
        self.pauli_randomize = pauli_randomize and self.split_inverse
        # always set pauli_randomize_barriers to False if pauli_randomize is False
        self.pauli_randomize_barriers = (
            pauli_randomize_barriers and self.pauli_randomize
        )
        self.middle_pauli_randomize = False

        self.left_and_right = left_and_right and pauli_randomize
        self.he = he

        if he and left_and_right:
            raise QiskitError("Not supported for HE and left and right")

        warnings.simplefilter("always")
        if self.split_inverse and len(qubits) % 2 == 1:
            self.split_inverse = False
            self.pauli_randomize = False
            self.middle_pauli_randomize = False
            self.left_and_right = False
            warnings.warn(
                "Cannot split and invert QV circuits with odd depth. Circuits will not "
                + "undergo these modifications and target bitstrings will not be computed."
            )

        self._static_trans_circuits = None

    def dd_circuits(self) -> List[QuantumCircuit]:
        # run transpiler first
        self._transpiled_circuits()

        if self.backend is None:
            raise QiskitError("Can't run dd without backend specified")

        durations = InstructionDurations.from_backend(self.backend)

        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(durations),
                PadDynamicalDecoupling(
                    durations, dd_sequence, qubits=self._physical_qubits
                ),
            ]
        )

        self._static_trans_circuits = pm.run(self._static_trans_circuits)

        return self._static_trans_circuits

    def _transpiled_circuits(self, retranspile: bool = False) -> List[QuantumCircuit]:
        """Transpiled circuits

        Args:
            retranspile: If true will re call the transpiled circuits function. If false, return
            the transpiled circuits if they exist
        """

        if (not retranspile) and (self._static_trans_circuits is not None):
            return self._static_trans_circuits

        self._static_trans_circuits = super()._transpiled_circuits()

        # add the measurement now
        # NEED TO BE CAREFUL BECAUSE THE LAYOUT MAY HAVE PERMUTED
        for circ in self._static_trans_circuits:
            cregs = ClassicalRegister(self._num_qubits, name="c")
            # circ.measure_active()
            # qv_circ.measure_active()
            circ.add_register(cregs)
            circ.barrier(self._physical_qubits)
            for qi in range(self._num_qubits):
                circ.measure(circ.layout.final_index_layout()[qi], qi)

        return self._static_trans_circuits

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            trials (int): Optional, number of times to generate new Quantum Volume
                circuits and calculate their heavy output.
            seed (None or int or SeedSequence or BitGenerator or Generator): A seed
                used to initialize ``numpy.random.default_rng`` when generating circuits.
                The ``default_rng`` will be initialized with this seed value everytime
                :meth:`circuits` is called.
        """
        options = super()._default_experiment_options()

        options.trials = 100
        options.seed = None

        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of Quantum Volume circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        rng = default_rng(seed=self.experiment_options.seed)
        circuits = []
        depth = self._num_qubits

        # Note: the trials numbering in the metadata is starting from 1 for each new experiment run
        for trial in range(1, self.experiment_options.trials + 1):
            if self.he:
                # assume linear connectivity
                # but we could feed in a coupling map
                # copied from qiskit.circuit.library.QuantumVolume
                # and adopted for he
                name = "quantum_volume_he" + str([depth, depth]).replace(" ", "")
                qv_circ = QuantumCircuit(depth, name=name)
                unitary_seeds = rng.integers(low=1, high=1000, size=[depth, depth])

                for i in range(depth):
                    all_edges = [(i, i + 1) for i in range(depth - 1)]
                    selected_edges = []
                    while all_edges:
                        rand_edge = all_edges.pop(rng.integers(len(all_edges)))
                        selected_edges.append(rand_edge)
                        old_all_edges = all_edges[:]
                        all_edges = []
                        # only keep edges in all_edges that do not share a vertex with rand_edge
                        for edge in old_all_edges:
                            if rand_edge[0] not in edge and rand_edge[1] not in edge:
                                all_edges.append(edge)

                    for edge_i, edge in enumerate(selected_edges):
                        su4 = random_unitary(
                            4, seed=unitary_seeds[i][edge_i]
                        ).to_instruction()
                        su4.label = "su4_" + str(unitary_seeds[i][edge_i])
                        qv_circ.compose(su4, [edge[0], edge[1]], inplace=True)

            else:
                qv_circ = QuantumVolumeCircuit(depth, depth, seed=rng)
                qv_circ = qv_circ.decompose()
            if self.split_inverse and depth % 2 == 0:
                if self.left_and_right:
                    qv_circ, target, right_qv_circ, right_target = (
                        self.mirror_qv_circuit(
                            qv_circ,
                            pauli_randomize=self.pauli_randomize,
                            pauli_randomize_barriers=self.pauli_randomize_barriers,
                            middle_pauli_randomize=self.middle_pauli_randomize,
                            left_and_right=self.left_and_right,
                            seed=rng,
                        )
                    )
                else:
                    qv_circ, target = self.mirror_qv_circuit(
                        qv_circ,
                        pauli_randomize=self.pauli_randomize,
                        pauli_randomize_barriers=self.pauli_randomize_barriers,
                        middle_pauli_randomize=self.middle_pauli_randomize,
                        left_and_right=self.left_and_right,
                        seed=rng,
                    )
            # qv_circ.measure_active()
            # qv_circ.add_register(cregs)
            # qv_circ.barrier([i for i in range(depth)])
            # for qi in range(depth):
            #    qv_circ.measure(qi, qi)

            qv_circ.metadata = {
                "experiment_type": self._type,
                "depth": depth,
                "trial": trial,
                "qubits": self.physical_qubits,
                "is_mirror_circuit": self.split_inverse,
                "is_from_left_half": True,
            }
            if self.left_and_right:
                right_qv_circ.measure_active()
                right_qv_circ.metadata = {
                    "experiment_type": self._type,
                    "depth": depth,
                    "trial": trial,
                    "qubits": self.physical_qubits,
                    "is_mirror_circuit": self.split_inverse,
                    "is_from_left_half": False,
                }
            if self.split_inverse and depth % 2 == 0:
                qv_circ.metadata["target_bitstring"] = target
                if self.left_and_right:
                    right_qv_circ.metadata["target_bitstring"] = right_target
            else:
                qv_circ.metadata["target_bitstring"] = "1" * depth
            circuits.append(qv_circ)
            if self.left_and_right:
                circuits.append(right_qv_circ)
        return circuits

    def mirror_qv_circuit(
        self,
        qv_circ: QuantumCircuit,
        pauli_randomize: Optional[bool] = True,
        pauli_randomize_barriers: Optional[bool] = False,
        middle_pauli_randomize: Optional[bool] = False,
        left_and_right: Optional[bool] = False,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    ) -> QuantumCircuit:
        """Modify QV circuits by splitting, inverting, and composing and/or Pauli
        twirling

        Args:
            qv_circ: the circuit to modify
            pauli_randomize: if True, add layers of random Paulis to the beginning
                             and the end of the cicuit
            pauli_randomize_barriers: if True, add barriers before/after Pauli twirling
            middle_pauli_randomize: if True, add a layer of random Paulis between
                                    the mirrored halves of the circuit
            left_and_right: if True, generate circuits using both the left and righ halves of the
                            original circuit
            seed: seed for RNG

        Returns:
            A modifed QV circuit
        """
        depth = self._num_qubits

        dag = circuit_to_dag(qv_circ)
        subdags = []  # DAG circuits in the left half
        right_subdags = []  # DAG circuits in the right half
        for i, layer in enumerate(dag.layers()):
            if i < depth / 2:
                subdags.append(layer["graph"])
            else:
                if left_and_right:
                    right_subdags.append(layer["graph"])

        new_dag = dag.copy_empty_like()
        right_new_dag = dag.copy_empty_like()
        for subdag in subdags:
            new_dag.compose(subdag)
        if left_and_right:
            for subdag in right_subdags:
                right_new_dag.compose(subdag)

        new_qv_circ = dag_to_circuit(new_dag)

        new_qv_circ_inv = new_qv_circ.inverse()  # mirrored QV circuit from left half
        if left_and_right:
            right_new_qv_circ = dag_to_circuit(right_new_dag)
            right_new_qv_circ_inv = (
                right_new_qv_circ.inverse()
            )  # mirrored QV circuit from right half

        paulis = random_pauli_list(
            depth,
            size=6,
            seed=seed,
            phase=False,
        )

        if pauli_randomize:
            first_pauli_circ = QuantumCircuit(depth)
            first_pauli_circ.compose(paulis[0], front=True, inplace=True)
            if pauli_randomize_barriers:
                first_pauli_circ.barrier()
            new_qv_circ.compose(first_pauli_circ, front=True, inplace=True)
        new_qv_circ.barrier()
        if middle_pauli_randomize:
            new_qv_circ.compose(paulis[1], inplace=True)
            new_qv_circ.barrier()
        new_qv_circ.compose(new_qv_circ_inv, inplace=True)
        if pauli_randomize_barriers:
            new_qv_circ.barrier()
        if pauli_randomize:
            new_qv_circ.compose(paulis[2], inplace=True)

        if left_and_right:
            if pauli_randomize:
                right_new_qv_circ_inv.compose(paulis[3], front=True, inplace=True)
            right_new_qv_circ_inv.barrier()
            if middle_pauli_randomize:
                right_new_qv_circ_inv.compose(paulis[4], inplace=True)
                right_new_qv_circ_inv.barrier()
            right_new_qv_circ_inv.compose(right_new_qv_circ, inplace=True)
            if pauli_randomize:
                right_new_qv_circ_inv.compose(paulis[5], inplace=True)

        composed_pauli = paulis[0].compose(paulis[2])
        if pauli_randomize:
            target = "".join(["1" if x else "0" for x in composed_pauli.x[::-1]])
            if left_and_right:
                right_composed_pauli = paulis[3].compose(paulis[5])
                right_target = "".join(
                    ["1" if x else "0" for x in right_composed_pauli.x[::-1]]
                )
            else:
                right_target = "0" * self._num_qubits
        else:
            target = "0" * depth
            right_target = "0" * self._num_qubits

        return_tuple = (new_qv_circ, target)
        if left_and_right:
            return_tuple += (right_new_qv_circ_inv, right_target)

        return return_tuple
