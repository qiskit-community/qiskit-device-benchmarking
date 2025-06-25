# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
MCM-RB Experiment
"""

import cmath
import math
import warnings
from typing import Sequence, List, Dict, Iterator, Optional, Union, Tuple

import lmfit
import numpy as np
import qiskit_experiments.curve_analysis as curve
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy.random import Generator
from numpy.random.bit_generator import BitGenerator, SeedSequence
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    Measure,
    Delay,
    Gate,
)
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library import UGate, SXGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.quantum_info import random_clifford, Clifford
from qiskit.transpiler import (
    StagedPassManager,
    PassManager,
    Layout,
    CouplingMap,
    TransformationPass,
)
from qiskit.transpiler.passes import (
    SetLayout,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    ApplyLayout,
)
from qiskit_experiments.data_processing import (
    DataProcessor,
    MarginalizeCounts,
    Probability,
)
from qiskit_experiments.framework import (
    AnalysisResultData,
    BaseExperiment,
    BackendTiming,
    Options,
    ExperimentData,
)
from qiskit_experiments.database_service.device_component import Qubit
from qiskit_experiments.visualization import PlotStyle
from scipy.linalg import det
from qiskit_experiments.curve_analysis import ScatterTable
from qiskit_experiments.framework.containers import FigureType, ArtifactData
from qiskit_ibm_runtime.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    PadDynamicalDecoupling,
)
from qiskit_ibm_runtime.transpiler.passes.scheduling import (
    DynamicCircuitInstructionDurations,
)

from qiskit.circuit import Qubit as Qubit_qiskit


class McmRB(BaseExperiment):
    """Mid-circuit measurement Randomized Benchmarking.

    # section: overview

        A mid-circuit measurement benchmarking suite developed from the ubiquitous paradigm of
        randomized benchmarking. The benchmarking suite can be used to both detect and quantify
        errors on both measured and spectator qubits.

    # section: reference
        .. ref_arxiv:: 1 2207.04836

    """

    def __init__(
        self,
        clif_qubit_sets: Sequence[Sequence[int]],
        meas_qubit_sets: Sequence[Sequence[int]],
        lengths: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        cliff_per_meas: int = 1,
        scheduling_method: Optional[str] = "alap",
        dd_sequence: Optional[Sequence[Gate]] = None,
        staggered_dd: Optional[bool] = True,
        dd_spacing_a: Optional[Sequence[float]] = None,
        dd_spacing_b: Optional[Sequence[float]] = None,
    ):
        """Create new MCM RB suite.

        Args:
            clif_qubit_sets: Qubits that standard RB sequence is applied to. List of lists.
            meas_qubit_sets: Qubits that mid-circuit measurement is applied to. List of lists. Same length as
            clif_qubit_sets. The same index in these lists is a grouping.
            lengths: Optional. List of number of Clifford elements.
            backend: Optional. Backend to run experiment.
            num_samples: Number of random circuit per Clifford length.
            seed: Optional. Random number generator instance or integer to set.
            cliff_per_meas: number of cliffords before a measurement
            scheduling_method: alap or asap (default: alap)
            staggered_dd: stagger DD pulse spacings between nearest neighbors (default: True)
            dd_spacing_a: pulse spacing for the first qubit set (default: [d/2, d, d, ..., d/2]). Sum must equal 1
            dd_spacing_b: pulse spacing for the second qubit set (default: [d, d, d, ..., 0]).
                          Sum must equal 1. Enabled with staggered_dd = True
        """

        if len(clif_qubit_sets) != len(meas_qubit_sets):
            raise QiskitError("Number of clif qubit sets must equal meas qubit sets")

        # make a flat version
        # this will be used for circuit generation
        clif_qubits = [y for x in clif_qubit_sets for y in x]
        meas_qubits = [y for x in meas_qubit_sets for y in x]

        if set(clif_qubits) & set(meas_qubits):
            raise QiskitError(
                "At least one of RB qubits is interleaved with the mid circuit measurement. "
                "This will generate invalid sequence. Please choose different configuration."
            )

        super().__init__(
            clif_qubits + meas_qubits,
            analysis=McmRBAnalysis.from_qubits(clif_qubit_sets, meas_qubit_sets),
            backend=backend,
        )
        self._clif_index = clif_qubits
        self._meas_index = meas_qubits
        self._clif_index_sets = clif_qubit_sets
        self._meas_index_sets = meas_qubit_sets
        self._cliff_per_meas = cliff_per_meas

        self.set_experiment_options(
            lengths=lengths,
            num_samples=num_samples,
            seed=seed,
            scheduling_method=scheduling_method,
            dd_sequence=dd_sequence,
            staggered_dd=staggered_dd,
            dd_spacing_a=dd_spacing_a,
            dd_spacing_b=dd_spacing_b,
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            lengths (Sequence[int]): A list of number of Clifford elements.
            min_length (int): The minimum number of Clifford elements in the benchmark circuits.
            max_length (int): The maximum number of Clifford elements in the benchmark circuits.
            num_length (int): The number of different Clifford element numbers to scan.
                The experiment automatically creates durations of linear increment
                along with ``min_length`` and ``max_length`` when user doesn't
                explicitly provide ``lengths``.
            num_samples (int): Number of samples, i.e. generated circuits, per Clifford number.
            seed (int): A seed used to generate random Clifford sequences.
            cliff_per_meas (int): The number of cliffords to do before a
            scheduling_method (str): `alap` or `asap`.
            dd_sequence (Sequence[Gate]): A list of Gate in the DD sequence.
            staggered_dd (bool): stagger DD pulse spacings between nearest neighbors if True
            dd_spacing_a (Sequence[float]): pulse spacing for the first qubit set
            dd_spacing_b (Sequence[float]): pulse spacing for the second qubit set
        """
        options = super()._default_experiment_options()
        options.update_options(
            lengths=None,
            min_length=1,
            max_length=150,
            num_length=15,
            num_samples=None,
            seed=None,
            cliff_per_meas=1,
            scheduling_method="alap",
            dd_sequence=None,
            staggered_dd=True,
            dd_spacing_a=None,
            dd_spacing_b=None,
        )

        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        opt = self.experiment_options
        rng = np.random.default_rng(seed=opt.seed)
        timing = BackendTiming(self.backend)
        meas_delay = timing.round_delay(time=self._get_mcm_duration())
        clif_delay = timing.round_delay(time=self._get_clifford_duration())

        if opt.lengths is None:
            lengths = np.linspace(
                opt.min_length, opt.max_length, opt.num_length, dtype=int
            )
        else:
            lengths = opt.lengths

        qregs = QuantumRegister(self.num_qubits, name="q")
        cregs = ClassicalRegister(self.num_qubits, name="c")
        exp_circs = []
        # Construct repeated measurement (or delay) layers
        meas_layer = QuantumCircuit(qregs, cregs)
        delay_layer = QuantumCircuit(qregs, cregs)
        for qind in self._meas_index:
            qreg = qregs[self.physical_qubits.index(qind)]
            creg = cregs[self.physical_qubits.index(qind)]
            meas_layer._append(Measure(), [qreg], [creg])
        for qind in self._clif_index:
            # Add a nominal delay to mark the qubit as non-idle.
            # The value is irrelevant, as the circuit is scheduled in the next step
            qreg = qregs[self.physical_qubits.index(qind)]
            meas_layer._append(Delay(meas_delay, timing.delay_unit), [qreg], [])
        # delay on all qubits for delay_layer
        for qreg in qregs:
            delay_layer._append(Delay(meas_delay, timing.delay_unit), [qreg], [])
        # Transpile layers including scheduling and DD
        meas_layer = self._transpile_single_circuit(
            meas_layer, apply_scheduling=self.experiment_options.dd_sequence is not None
        )
        delay_layer = self._transpile_single_circuit(
            delay_layer,
            apply_scheduling=self.experiment_options.dd_sequence is not None,
        )
        # remove idle qubits so we can map the transpiled layers to the
        # original circuit
        pm = PassManager(RemoveIdleQubits())
        meas_layer = pm.run(meas_layer)
        delay_layer = pm.run(delay_layer)
        for _ in range(opt.num_samples):
            for length in lengths:
                generators = (
                    self._generate_sequences(length * self._cliff_per_meas, rng)
                    for _ in self._clif_index
                )

                # Generate MCM RB circuit
                mcm_circ = QuantumCircuit(qregs, cregs)
                mcm_circ.metadata = {
                    "xval": length,
                    "type": "mcm",
                }
                del_circ = QuantumCircuit(qregs, cregs)
                del_circ.metadata = {
                    "xval": length,
                    "type": "del",
                }
                rep_circ = QuantumCircuit(qregs, cregs)
                rep_circ.metadata = {
                    "xval": length,
                    "type": "rep",
                }
                n_elms = 0
                for elms in zip(*generators):
                    n_elms += 1
                    for qind, elm in zip(self._clif_index, elms):
                        qreg = qregs[self.physical_qubits.index(qind)]
                        # Add a single random clifford
                        for inst in self._sequence_to_instructions(elm):
                            mcm_circ._append(inst, [qreg], [])
                            del_circ._append(inst, [qreg], [])
                        # Add Delay of equivalent length
                        rep_circ._append(
                            Delay(clif_delay, timing.delay_unit), [qreg], []
                        )
                    # Sync time
                    mcm_circ.barrier()
                    del_circ.barrier()
                    rep_circ.barrier()
                    if n_elms <= (length * self._cliff_per_meas) and (
                        np.mod(n_elms, self._cliff_per_meas) == 0
                    ):
                        qregs_mapped = [
                            qregs[self.physical_qubits.index(mlq._index)]
                            for mlq in meas_layer.qubits
                        ]
                        # Interleave MCM
                        mcm_circ.compose(meas_layer, inplace=True, qubits=qregs_mapped)
                        rep_circ.compose(meas_layer, inplace=True, qubits=qregs_mapped)
                        # Interleave Delay of equivalent length
                        del_circ.compose(delay_layer, inplace=True, qubits=qregs_mapped)
                        # Sync time
                        mcm_circ.barrier()
                        del_circ.barrier()
                        rep_circ.barrier()
                for qubit, clbit in zip(qregs, cregs):
                    mcm_circ._append(Measure(), [qubit], [clbit])
                    del_circ._append(Measure(), [qubit], [clbit])
                    rep_circ._append(Measure(), [qubit], [clbit])

                exp_circs.extend([mcm_circ, del_circ, rep_circ])

        return exp_circs

    def _generate_sequences(self, length: int, rng: Generator) -> Iterator[Clifford]:
        """Generate N+1 Clifford sequences with inverse at the end."""
        composed = Clifford([[1, 0], [0, 1]])
        for _ in range(length):
            elm = random_clifford(1, rng)
            composed = composed.compose(elm)
            yield elm
        if length > 0:
            yield composed.adjoint()

    def _get_mcm_duration(self) -> float:
        """Get mid-circuit measurement duration from backend."""
        if not self.backend:
            # This is virtual circuit and all instruction is zero duration.
            return 0

        mcm_durs = []
        for index in self._meas_index:
            dur = self.backend.properties().readout_length(index)
            mcm_durs.append(dur)
        return max(mcm_durs)

    def _get_clifford_duration(self) -> float:
        """Get decomposed Clifford gate duration from backend."""
        if not self.backend:
            # This is virtual circuit and all instruction is zero duration.
            return 0

        basis_gates = self.backend.configuration().basis_gates
        if all(op in basis_gates for op in ("sx", "rz")):
            clifford_durs = []
            for index in self._clif_index:
                # All Clifford can be decomposed with 2 physical SX gates
                dur = 2 * self.backend.properties().gate_length("sx", index)
                clifford_durs.append(dur)
            return max(clifford_durs)
        if "u" in basis_gates:
            clifford_durs = []
            for index in self._clif_index:
                dur = self.backend.properties().gate_length("u", index)
                clifford_durs.append(dur)
            return max(clifford_durs)
        raise QiskitError(
            f"Current decomposition mechanism doesn't support basis gates {basis_gates}."
        )

    def _sequence_to_instructions(self, elm: Clifford) -> List[Instruction]:
        """Single qubit Clifford decomposition with fixed number of physical gates.

        This overrules standard Qiskit transpile protocol and immediately
        apply hard-coded decomposition with respect to the backend basis gates.
        Note that this decomposition ignores global phase.

        This decomposition guarantees constant gate duration per every Clifford.
        """
        if not self.backend:
            return [elm.to_instruction()]
        else:
            basis_gates = self.backend.configuration().basis_gates
            # First decompose into Euler angle rotations.
            theta, phi, lam = self._zyz_decomposition(elm.to_matrix())

            if all(op in basis_gates for op in ("sx", "rz")):
                return [
                    RZGate(lam),
                    SXGate(),
                    RZGate(theta + math.pi),
                    SXGate(),
                    RZGate(phi - math.pi),
                ]
            if "u" in basis_gates:
                return [UGate(theta, phi, lam)]
        raise QiskitError(
            f"Current decomposition mechanism doesn't support basis gates {basis_gates}."
        )

    def _zyz_decomposition(self, mat: np.ndarray):
        # This code is copied from
        # qiskit.quantum_info.synthesis.one_qubit_decompose.OneQubitEulerDecomposer
        su_mat = det(mat) ** (-0.5) * mat
        theta = 2 * math.atan2(abs(su_mat[1, 0]), abs(su_mat[0, 0]))
        phiplambda2 = cmath.phase(su_mat[1, 1])
        phimlambda2 = cmath.phase(su_mat[1, 0])
        phi = phiplambda2 + phimlambda2
        lam = phiplambda2 - phimlambda2

        return theta, phi, lam

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Apply layout and add calibrations."""

        if any(self._set_transpile_options):
            warnings.warn(
                "Transpile of this experiment is hard-coded. "
                "Provided transpile options are all ignored.",
                UserWarning,
            )

        return list(map(self._transpile_single_circuit, self.circuits()))

    def _transpile_single_circuit(self, circuit, apply_scheduling=False):
        initial_layout = Layout.from_intlist(self.physical_qubits, *circuit.qregs)
        coupling_map = CouplingMap(self._backend_data.coupling_map)

        transpiler = StagedPassManager(stages=["layout", "calibration", "scheduling"])
        transpiler.layout = PassManager(
            [
                SetLayout(initial_layout),
                FullAncillaAllocation(coupling_map),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ]
        )
        transpiler.scheduling = (
            self._generate_dd_pass_manager() if apply_scheduling else None
        )

        return transpiler.run(circuit)

    def _metadata(self) -> Dict[str, any]:
        """Return experiment metadata for ExperimentData."""
        metadata = super()._metadata()
        metadata["qubit_slot_map"] = {
            qind: sind for sind, qind in enumerate(self.physical_qubits)
        }
        metadata["clif_qubits"] = self._clif_index
        metadata["meas_qubits"] = self._meas_index
        metadata["cliff_per_meas"] = self._cliff_per_meas

        return metadata

    def _generate_dd_pass_manager(self) -> PassManager:
        """Generate a pass manager for scheduling and dynamical decoupling"""
        options = self.experiment_options
        instruction_durations = DynamicCircuitInstructionDurations.from_backend(
            self.backend
        )
        pulse_alignment = self.backend.target.pulse_alignment
        coupling_map = self.backend.coupling_map

        dd_pass = PadDynamicalDecoupling(
            instruction_durations,
            options.dd_sequence,
            pulse_alignment=pulse_alignment,
            spacings=options.dd_spacing_a,
            coupling_map=coupling_map if options.staggered_dd else None,
            alt_spacings=options.dd_spacing_b,
            skip_reset_qubits=False,
            # unused options:
            # extra_slack_distribution=options.extra_slack_distribution,
            # sequence_min_length_ratios=options.sequence_min_length_ratios,
            # insert_multiple_cycles=options.insert_multiple_cycles,
        )

        if options.scheduling_method == "alap":
            timing_pass_cls = ALAPScheduleAnalysis
        elif options.scheduling_method == "asap":
            timing_pass_cls = ASAPScheduleAnalysis
        else:
            raise ValueError(
                f"Invalid timing strategy provided: {options.scheduling_method}"
            )

        return PassManager([timing_pass_cls(instruction_durations), dd_pass])


class SubDecayFit(curve.CurveAnalysis):
    """RB decay fit for single qubit."""

    def __init__(self, name: Optional[str] = None):
        """Create new decay fit analysis.

        Args:
            name: Name of this fitting.
        """
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="a * alpha ** x + b",
                    name="rb",
                )
            ],
            name=name,
        )

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.ScatterTable,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        user_opt.bounds.set_if_empty(
            a=(0, 1),
            alpha=(0, 1),
            b=(0, 1),
        )

        b_guess = 1 / 2
        alpha_guess = curve.guess.rb_decay(curve_data.x, curve_data.y, b=b_guess)
        a_guess = (curve_data.y[0] - b_guess) / (alpha_guess ** curve_data.x[0])

        user_opt.p0.set_if_empty(
            b=b_guess,
            a=a_guess,
            alpha=alpha_guess,
        )

        return user_opt


class McmRBAnalysis(curve.CompositeCurveAnalysis):
    """Mid-circuit measurement RB analysis."""

    def __init__(self, analyses, q_sets):
        """Create new decay fit analysis.

        Args:
            name: Name of this fitting.
        """
        self._q_sets = q_sets
        super().__init__(analyses)

    @classmethod
    def from_qubits(
        cls,
        clif_qubit_sets: Sequence[Sequence[int]],
        meas_qubit_sets: Sequence[Sequence[int]],
    ) -> "McmRBAnalysis":
        """Create McmRBAnalysis from index list of Clifford and MCM qubits.

        Args:
            clif_qubit_sets: Qubits that standard RB sequence is applied to.
            meas_qubit_sets: Qubits that mid-circuit measurement is applied to.
            start_slot_q: slot to start counting from for the clif qubits
            start_slot_m: slot to start counting from for the meas qubits

        Returns:
            McmRBAnalysis instance.
        """

        clif_start_ind = [
            int(np.sum([len(x) for x in clif_qubit_sets[0:i]]))
            for i in range(len(clif_qubit_sets))
        ]
        meas_start_ind = [
            int(
                np.sum([len(x) for x in clif_qubit_sets])
                + np.sum([len(x) for x in meas_qubit_sets[0:i]])
            )
            for i in range(len(clif_qubit_sets))
        ]

        analyses = []
        visualization_params = {}

        default_markers = MarkerStyle(".").filled_markers

        for i in range(len(clif_qubit_sets)):
            slot_idx = clif_start_ind[i]
            for qubit in clif_qubit_sets[i]:
                if 1:
                    for canv_ind, rb_type in enumerate(("mcm", "del", "rep")):
                        analyses.append(
                            cls._initialize_sub_analysis(
                                qubit, slot_idx, rb_type, "clif", i
                            )
                        )
                        mind = slot_idx % len(default_markers)
                        visualization_params[f"rb_{rb_type}_{qubit}_{i}"] = {
                            "canvas": canv_ind,
                            "color": "red",
                            "label": f"clif (Q{qubit})",
                            "symbol": default_markers[mind],
                        }
                slot_idx += 1
            slot_idx = meas_start_ind[i]
            if 1:
                for qubit in meas_qubit_sets[i]:
                    for canv_ind, rb_type in enumerate(("mcm", "del", "rep")):
                        analyses.append(
                            cls._initialize_sub_analysis(
                                qubit, slot_idx, rb_type, "meas", i
                            )
                        )
                        mind = slot_idx % len(default_markers)
                        visualization_params[f"rb_{rb_type}_{qubit}_{i}"] = {
                            "canvas": canv_ind,
                            "color": "blue",
                            "label": f"meas (Q{qubit})",
                            "symbol": default_markers[mind],
                        }
                    slot_idx += 1

        # Initialize Mcm analysis and apply visualization options.
        analysis = cls(analyses, clif_qubit_sets)
        analysis.plotter.set_figure_options(series_params=visualization_params)

        return analysis

    @classmethod
    def _initialize_sub_analysis(cls, qubit, slot, rb_type, qubit_type, set_ind):
        parameter_name_alpha = f"alpha_{rb_type}_{qubit_type}_q{qubit}"

        analysis = SubDecayFit(name=f"{rb_type}_{qubit}_{set_ind}")

        analysis.set_options(
            data_processor=DataProcessor(
                "counts", [MarginalizeCounts({slot}), Probability("0")]
            ),
            average_method="sample",
            filter_data={"type": rb_type},
            result_parameters=[curve.ParameterRepr("alpha", parameter_name_alpha)],
        )
        return analysis

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plotter.set_options(
            subplots=(3, 1),
            style=PlotStyle(
                {
                    "figsize": (8, 10),
                    "legend_loc": "lower right",
                    "textbox_rel_pos": (0.28, -0.10),
                }
            ),
        )
        options.plotter.set_figure_options(
            xlabel="Number of Measurement Layers",
            ylabel=["MCM-P(0)", "DEL-P(0)", "REP-P(0)"],
            ylim=(0.45, 1.05),
        )
        return options

    def _create_analysis_results(
        self,
        fit_data: Dict[str, curve.CurveFitResult],
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Compute IRB estimates from MCM referenced by DEL RB."""

        outcomes = super()._create_analysis_results(fit_data, quality, **metadata)

        mcm_rb_fits = {}
        del_rb_fits = {}
        rep_rb_fits = {}
        qubits = set()
        for key, data in fit_data.items():
            rb_type, qind, setind = key.split("_")
            qind = int(qind)
            if rb_type == "mcm":
                mcm_rb_fits[qind] = data
            elif rb_type == "del":
                del_rb_fits[qind] = data
            elif rb_type == "rep":
                rep_rb_fits[qind] = data
            else:
                continue
            qubits.add(qind)

        for qind in qubits:
            qset_ind = -1
            for i in range(len(self._q_sets)):
                if qind in self._q_sets[i]:
                    qset_ind = i

            if qset_ind < 0:
                # measurement qubit -> don't report
                continue

            mcm_rb_fit = mcm_rb_fits.get(qind, None)
            del_rb_fit = del_rb_fits.get(qind, None)
            if mcm_rb_fit and del_rb_fit:
                mcm_alpha = mcm_rb_fit.ufloat_params["alpha"]
                del_alpha = del_rb_fit.ufloat_params["alpha"]
                epsilon = (1 - mcm_alpha / del_alpha) / 2
                outcomes.append(
                    AnalysisResultData(
                        name="mcm_rb_err",
                        value=epsilon,
                        quality=quality,
                        extra=metadata,
                        device_components=[Qubit(qind)],
                    )
                )

        for qind in qubits:
            qset_ind = -1
            for i in range(len(self._q_sets)):
                if qind in self._q_sets[i]:
                    qset_ind = i

            if qset_ind >= 0:
                # measurement qubit
                continue

            rep_rb_fit = rep_rb_fits.get(qind, None)

            if rep_rb_fit:
                p0 = rep_rb_fit.ufloat_params["a"] + rep_rb_fit.ufloat_params["b"]
                p100 = (
                    rep_rb_fit.ufloat_params["a"]
                    * rep_rb_fit.ufloat_params["alpha"] ** 100
                    + rep_rb_fit.ufloat_params["b"]
                )

                err100 = p100 / p0

                outcomes.append(
                    AnalysisResultData(
                        name="qnd_err_100rep_drop",
                        value=err100,
                        quality=quality,
                        extra=metadata,
                        device_components=[Qubit(qind)],
                    )
                )

        return outcomes

    def _create_figures(
        self,
        curve_data: ScatterTable,
    ) -> List[Figure]:
        fig_list = []

        self.plotter.clear_supplementary_data()

        for i in range(len(self._q_sets)):
            self.plotter.clear_series_data()
            for analysis in self.analyses():
                if int(analysis.name.split("_")[2]) != i:
                    continue

                group_data = curve_data.filter(analysis=analysis.name)
                model_names = analysis.model_names()
                for series_id, sub_data in group_data.iter_by_series_id():
                    full_name = f"{model_names[series_id]}_{analysis.name}"
                    # Plot raw data scatters
                    if analysis.options.plot_raw_data:
                        raw_data = sub_data.filter(category="raw")
                        self.plotter.set_series_data(
                            series_name=full_name,
                            x=raw_data.x,
                            y=raw_data.y,
                        )
                    # Plot formatted data scatters
                    formatted_data = sub_data.filter(
                        category=analysis.options.fit_category
                    )
                    self.plotter.set_series_data(
                        series_name=full_name,
                        x_formatted=formatted_data.x,
                        y_formatted=formatted_data.y,
                        y_formatted_err=formatted_data.y_err,
                    )
                    # Plot fit lines
                    line_data = sub_data.filter(category="fitted")
                    if len(line_data) == 0:
                        continue
                    fit_stdev = line_data.y_err
                    self.plotter.set_series_data(
                        series_name=full_name,
                        x_interp=line_data.x,
                        y_interp=line_data.y,
                        y_interp_err=fit_stdev
                        if np.isfinite(fit_stdev).all()
                        else None,
                    )

            fig_list.append(self.plotter.figure())

        return fig_list

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[Union[AnalysisResultData, ArtifactData]], List[FigureType]]:
        result_data: List[Union[AnalysisResultData, ArtifactData]] = []
        figures: List[FigureType] = []

        result_data, figures = super()._run_analysis(experiment_data)

        artdata = None
        for result in result_data:
            if isinstance(result, ArtifactData):
                if result.name == "fit_summary":
                    artdata = result.data
                    break

        if (artdata is not None) and (not all([artdata[i].success for i in artdata])):
            # run the analysis anyway

            total_quality = super()._evaluate_quality(artdata)

            composite_results = self._create_analysis_results(
                fit_data=artdata, quality=total_quality, **super().options.extra.copy()
            )
            result_data.extend(composite_results)

        return result_data, figures


class RemoveIdleQubits(TransformationPass):
    """Remove idle qubits from a circuit."""

    def run(self, dag):
        dag.remove_qubits(
            *(bit for bit in dag.idle_wires() if isinstance(bit, Qubit_qiskit))
        )
        return dag
