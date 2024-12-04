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
import numpy as np
from numpy import pi
from numpy.random import Generator, BitGenerator, SeedSequence
from scipy.stats import entropy
from uncertainties import unumpy as unp
from scipy.spatial.distance import hamming

from qiskit.circuit import Instruction
from qiskit.providers.backend import Backend
from qiskit.circuit.library import CXGate

from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.data_processing import DataProcessor

from .mirror_rb_experiment import MirrorRB, MirrorRBAnalysis
from .mirror_qv_analysis import _ComputeQuantities


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
        initial_entangling_angle: float = pi/2,
        final_entangling_angle: float = 0,
        analyzed_quantity: str = "Effective Polarization",
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
            two_qubit_gate=CXGate(),
            num_samples=num_samples,
            sampler_opts=sampler_opts,
            seed=seed,
            inverting_pauli_layer=inverting_pauli_layer,
            full_sampling=False,
            start_end_clifford=False,
            initial_entangling_angle = initial_entangling_angle,
            final_entangling_angle = final_entangling_angle,
        )

        self.analysis = MirrorQAAnalysis()

class MirrorQAAnalysis(MirrorRBAnalysis):

    @classmethod
    def _default_options(cls):
        default_options = super()._default_options()

        default_options.set_validator(
            field="analyzed_quantity",
            validator_value=[
                "Success Probability",
                "Adjusted Success Probability",
                "Effective Polarization",
                "Mutual Information"
            ],
        )
        return default_options

    def _initialize(self, experiment_data: ExperimentData):
        """Initialize curve analysis by setting up the data processor for Mirror
        RB data.

        Args:
            experiment_data: Experiment data to analyze.
        """
        super()._initialize(experiment_data)

        num_qubits = len(self._physical_qubits)
        target_bs = []
        pairs = []
        singles = []
        for circ_result in experiment_data.data():
            pairs.append(circ_result["metadata"]["pairs"])
            singles.append(circ_result["metadata"]["singles"])
            if circ_result["metadata"]["inverting_pauli_layer"] is True:
                target_bs.append("0" * num_qubits)
            else:
                target_bs.append(circ_result["metadata"]["target"])

        self.set_options(
            data_processor=DataProcessor(
                input_key="counts",
                data_actions=[
                    _ComputeQAQuantities(
                        analyzed_quantity=self.options.analyzed_quantity,
                        num_qubits=num_qubits,
                        target_bs=target_bs,
                        pairs=pairs,
                        singles=singles,
                        coupling_map=circ_result["metadata"]["coupling_map"],
                    )
                ],
            )
        )

class _ComputeQAQuantities(_ComputeQuantities):
    """Data processing node for computing useful mirror RB quantities from raw results."""

    def __init__(
        self,
        num_qubits,
        target_bs,
        pairs,
        singles,
        coupling_map,
        analyzed_quantity: str = "Effective Polarization",
        validate: bool = True,
    ):
        """
        Args:
            num_qubits: Number of qubits.
            quantity: The quantity to calculate.
            validate: If set to False the DataAction will not validate its input.
        """
        super().__init__(
            num_qubits = num_qubits,
            target_bs = target_bs,
            pairs = pairs,
            singles = singles,
            analyzed_quantity = analyzed_quantity,
            validate = validate,
        )
        self._coupling_map = coupling_map

    def _process(self, data: np.ndarray):
        if self._analyzed_quantity == "Mutual Information":
            qa = QuantumAwesomeness(self._coupling_map)
            mutual_infos = qa.mean_mutual_info(data,self._pairs)
            y_data = []
            y_data_unc = []
            for mi in mutual_infos['paired']:
                y_data.append(mi)
                y_data_unc.append(0)
            return unp.uarray(y_data, y_data_unc)
        else:
            return super()._process(data)

class QuantumAwesomeness():
    def __init__(
            self,
            coupling_map
    ):
        self._coupling_map= coupling_map

    def mutual_info(self, data: np.ndarray):

        mutual_infos = []
        for circ_data in data:
            if 'counts' not in circ_data:
                counts = circ_data
            else:
                counts = circ_data['counts']
            shots = sum(counts.values())
            p = {}
            for j,k in self._coupling_map:
                p[j,k] = {'00':0, '01':0, '10':0, '11':0}
                for string in counts:
                    ss = string[-1-j] + string[-1-k]
                    p[j,k][ss] += counts[string]
                for ss in p[j,k]:
                    p[j,k][ss] /= shots

            mi = {}
            for j,k in self._coupling_map:
                if j<k:
                    ps_l = [p[j, k][b+'0']+p[j, k][b+'1'] for b in ['0', '1']]
                    ps_r = [p[j, k]['0'+b]+p[j, k]['1'+b] for b in ['0', '1']]
                    mi[j, k] = - entropy(list(p[j,k].values()), base=2)
                    for ps in [ps_l, ps_r]:
                        mi[j,k] += entropy(ps, base=2)
            mutual_infos.append(mi)
        
        return mutual_infos
            
    def mean_mutual_info(self, data: np.ndarray, pairs):
        mutual_infos = self.mutual_info(data)
        mean_mi = {'paired':[], 'single':[]}
        for c, mi in enumerate(mutual_infos):
            mean_mi['paired'].append([])
            mean_mi['single'].append([])
            for pair, value in mi.items():
                if tuple(pair) in pairs[c] or tuple(pair[::-1]) in pairs[c]:
                    mean_mi['paired'][-1].append(value)
                else:
                    mean_mi['single'][-1].append(value)
            for ps in mean_mi:
                if mean_mi[ps][-1]:
                    mean_mi[ps][-1] = np.mean(mean_mi[ps][-1])
                else:
                    mean_mi[ps][-1] = np.nan
        return mean_mi