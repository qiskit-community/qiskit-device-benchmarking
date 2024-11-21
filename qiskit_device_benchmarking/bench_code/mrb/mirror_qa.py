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
Mirror QA Experiment class.
"""
from typing import Union, Iterable, Optional, List, Sequence
from numpy import pi
from numpy.random import Generator, BitGenerator, SeedSequence
import rustworkx as rx

from qiskit.circuit import Instruction
from qiskit.providers.backend import Backend
from qiskit.circuit.library import CXGate

from .mirror_rb_experiment import MirrorRB


class MirrorQA(MirrorRB):
    """An experiment to measure gate infidelity using mirrored circuit
    layers sampled from a defined distribution.

    # section: overview
        Mirror randomized benchmarking (mirror RB) estimates the average error rate of
        quantum gates using layers of gates sampled from a distribution that are then
        inverted in the second half of the circuit.

        The default mirror RB experiment generates circuits of layers of Cliffords,
        consisting of single-qubit Cliffords and a two-qubit gate such as CX,
        interleaved with layers of Pauli gates and capped at the start and end by a
        layer of single-qubit Cliffords. The second half of the Clifford layers are the
        inverses of the first half of Clifford layers. This algorithm has a lot less
        overhead than the standard randomized benchmarking, which requires
        n-qubit Clifford gates, and so it can be used for benchmarking gates on
        10s of or even 100+ noisy qubits.

        After running the circuits on a backend, various quantities (success
        probability, adjusted success probability, and effective polarization)
        are computed and used to fit an exponential decay curve and calculate
        the EPC (error per Clifford, also referred to as the average gate
        infidelity) and entanglement infidelity (see references for more info).

    # section: analysis_ref
        :class:`MirrorRBAnalysis`

    # section: manual
        :doc:`/manuals/verification/mirror_rb`

    # section: reference
        .. ref_arxiv:: 1 2112.09853
        .. ref_arxiv:: 2 2008.11294
        .. ref_arxiv:: 3 2204.07568

    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        physical_qubits: Sequence[int],
        lengths: Iterable[int],
        pauli_randomize: bool = True,
        sampling_algorithm: str = "edge_grab",
        two_qubit_gate_density: float = 0.25,
        two_qubit_gate: Instruction = CXGate(),
        num_samples: int = 3,
        sampler_opts: Optional[dict] = {},
        backend: Optional[Backend] = None,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        inverting_pauli_layer: bool = False,
    ):
        """Initialize a mirror quantum awesomeness experiment.

        Args:
            physical_qubits: A list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            sampling_algorithm: The sampling algorithm to use for generating
                circuit layers. Defaults to "edge_grab" which uses :class:`.EdgeGrabSampler`.
            start_end_clifford: If True, begin the circuit with uniformly random 1-qubit
                Cliffords and end the circuit with their inverses.
            pauli_randomize: If True, surround each sampled circuit layer with layers of
                uniformly random 1-qubit Paulis.
            two_qubit_gate_density: Expected proportion of qubit sites with two-qubit
                gates over all circuit layers (not counting optional layers at the start
                and end). Only has effect if the default sampler
                :class:`.EdgeGrabSampler` is used.
            two_qubit_gate: The two-qubit gate to use. Defaults to
                :class:`~qiskit.circuit.library.CXGate`. Only has effect if the
                default sampler :class:`.EdgeGrabSampler` is used.
            num_samples: Number of samples to generate for each sequence length.
            sampler_opts: Optional dictionary of keyword arguments to pass to the sampler.
            backend: Optional, the backend to run the experiment on.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                when generating circuits. The ``default_rng`` will be initialized
                with this seed value every time :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                all lengths. If False for sample of lengths longer sequences are
                constructed by appending additional Clifford samples to shorter
                sequences.
            inverting_pauli_layer: If True, a layer of Pauli gates is appended at the
                end of the circuit to set all qubits to 0.

        Raises:
            QiskitError: if an odd length or a negative two qubit gate density is provided
        """

        super().__init__(
            physical_qubits,
            lengths,
            backend=backend,
            pauli_randomize=pauli_randomize,
            sampling_algorithm=sampling_algorithm,
            two_qubit_gate_density=two_qubit_gate_density,
            two_qubit_gate=two_qubit_gate,
            num_samples=num_samples,
            sampler_opts=sampler_opts,
            seed=seed,
            inverting_pauli_layer=inverting_pauli_layer,
            full_sampling=False,
            start_end_clifford=False,
            initial_entangling_angle = pi/2,
            final_entangling_angle = pi/2,
        )