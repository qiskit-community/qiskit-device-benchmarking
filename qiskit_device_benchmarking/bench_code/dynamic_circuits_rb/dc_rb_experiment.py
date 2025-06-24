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
Dynamic circuits RB Experiment class.
"""

from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
import cmath
import math

from qiskit_ibm_runtime.transpiler.passes.scheduling import (
    DynamicCircuitInstructionDurations,
)
from scipy.linalg import det
from numpy.random import default_rng
from qiskit.circuit import QuantumCircuit, Delay

from qiskit.quantum_info import Clifford
from qiskit.quantum_info.random import random_clifford
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library import UGate, SXGate, RZGate
from qiskit.exceptions import QiskitError
from numpy.random import Generator
from typing import Sequence, List, Iterator
import numpy as np
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import (
    BaseExperiment,
    BackendTiming,
)
from qiskit_experiments.data_processing import (
    DataProcessor,
    Probability,
    MarginalizeCounts,
)
import qiskit_experiments.curve_analysis as curve
from ..mcm_rb import SubDecayFit


class DynamicCircuitsRB(BaseExperiment):
    """Dynamic circuits Randomized Benchmarking.

    # section: overview

        a series of dynamic circuit benchmarking routines based on interleaving dynamic circuit
        operation blocks in one-qubit randomized benchmarking sequences of data qubits. The blocks span
        between the set of data qubits and a measurement qubit and may include feedforward operations
        based on the measurement.

    # section: reference
        .. ref_arxiv:: 1 2408.07677

    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Backend,
        n_blocks=(0, 1, 2, 3, 4, 5, 10, 15, 20),
        num_samples=3,
        seed=100,
        cliff_per_meas=5,
        ff_operations=("I_c0", "Z_c0", "I_c1", "Z_c1", "Delay"),
        ff_delay=2120,
        plot_measured_qubit=False,
        plot_summary=False,
    ):
        """Dynamic circuits RB.
        Args:
            physical_qubits: The qubits on which to run the experiment.
            backend: The backend to run the experiment on.
            n_blocks: Number of measurements/feedforward operations
            num_samples: Number of different sequences to generate.
            seed: Seed for the random number generator.
            ff_operations: Sequence of the dynamic circuits blocks labels.
            ff_delay: Feedforward latency in dt units.
            plot_measured_qubit: Plot the decay curve of the measured qubit.
            plot_summary: Plot summary of the decay parameters.
        """
        super().__init__(physical_qubits=physical_qubits, backend=backend)
        self.analysis = DynamicCircuitsRBAnalysis(
            physical_qubits=physical_qubits,
            ff_operations=ff_operations,
            plot_measured_qubit=plot_measured_qubit,
            plot_summary=plot_summary,
        )
        self.n_blocks = n_blocks
        self.seed = seed
        self.num_samples = num_samples
        self.ff_operations = ff_operations
        self.ff_delay = ff_delay
        self.cliff_per_meas = cliff_per_meas
        if "H_CNOT" in self.ff_operations and len(physical_qubits) != 2:
            raise Exception("The CNOT blocks are supported only for 2 physical qubits")

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        rng = default_rng(seed=self.seed)

        n_qubits = self.num_qubits

        # Construct interleaved parts
        ff_circs = []
        for ff_type in self.ff_operations:
            ff_circs.append(self.ff_circ(ff_type))

        circuits = []

        for i in range(self.num_samples):
            for length in self.n_blocks:
                generators = (
                    self._generate_sequences(length * self.cliff_per_meas, rng)
                    for _ in range(n_qubits - 1)
                )

                # Generate MCM RB circuit
                circs = []
                for ff_type in self.ff_operations:
                    circ = QuantumCircuit(n_qubits, n_qubits)
                    circ.metadata = {
                        "xval": length,
                        "physical_qubits": self.physical_qubits,
                        "num_sample": i,
                        "ff_type": ff_type,
                    }
                    circs.append(circ)

                n_elms = 0
                for elms in zip(*generators):
                    n_elms += 1
                    for q, elm in enumerate(elms):
                        # Add a single random clifford
                        for inst in self._sequence_to_instructions(elm):
                            for circ in circs:
                                circ._append(inst, [circ.qubits[q]], [])
                    # Sync time
                    for circ in circs:
                        circ.barrier()
                    if n_elms <= (length * self.cliff_per_meas) and (
                        np.mod(n_elms, self.cliff_per_meas) == 0
                    ):
                        # Interleave MCM
                        for circ, ff_circ in zip(circs, ff_circs):
                            circ.compose(ff_circ, inplace=True, qubits=circ.qubits)
                            circ.barrier()
                for circ in circs:
                    circ.barrier()
                    circ.measure(circ.qubits, circ.clbits)

                circuits.extend(circs)

        return circuits

    def ff_circ(self, ff_type):
        circ = QuantumCircuit(self.num_qubits, self.num_qubits)
        timing = BackendTiming(self.backend)
        durations = DynamicCircuitInstructionDurations.from_backend(self.backend)
        clbits = circ.clbits
        qubits = circ.qubits
        if ff_type == "H_CNOT":
            circ.h(qubits[-1])
            circ.barrier()
            circ.cx(qubits[-1], qubits[0])
            circ.barrier()
            circ.measure(qubits[-1], clbits[-1])
            with circ.if_test((clbits[-1], 1)):
                circ.x(qubits[0])
                circ.x(qubits[-1])
        elif ff_type == "H_CNOT_FFDD":
            meas_dt = durations.get("measure", 0, "dt")
            x_dt = durations.get("x", 0, "dt")
            ff_dt = self.ff_delay
            delay1 = timing.round_delay(
                time=((meas_dt - ff_dt - 2 * x_dt) / 2) * timing.dt
            )
            delay2 = timing.round_delay(time=(ff_dt - x_dt) * timing.dt)
            circ.h(qubits[-1])
            circ.barrier()
            circ.cx(qubits[-1], qubits[0])
            circ.barrier()
            circ.x([qubits[0]])
            circ.append(Delay(delay1, "dt"), [qubits[0]], [])
            circ.x([qubits[0]])
            circ.append(Delay(delay1, "dt"), [qubits[0]], [])
            circ.x([qubits[0]])
            circ.append(Delay(delay2, "dt"), [qubits[0]], [])
            circ.x([qubits[0]])
            circ.measure(qubits[-1], clbits[-1])
            circ.barrier()
            with circ.if_test((clbits[-1], 1)):
                circ.x(qubits[0])
                circ.x(qubits[-1])
        elif ff_type == "H_CNOT_MDD":
            meas_dt = durations.get("measure", 0, "dt")
            x_dt = durations.get("x", 0, "dt")
            delay_quarter = timing.round_delay(
                time=((meas_dt - 2 * x_dt) / 4) * timing.dt
            )
            circ.h(qubits[-1])
            circ.barrier()
            circ.cx(qubits[-1], qubits[0])
            circ.barrier()
            circ.barrier()
            circ.x([qubits[0]])
            circ.append(Delay(delay_quarter, "dt"), [qubits[0]], [])
            circ.x([qubits[0]])
            circ.append(Delay(delay_quarter * 2, "dt"), [qubits[0]], [])
            circ.x([qubits[0]])
            circ.append(Delay(delay_quarter, "dt"), [qubits[0]], [])
            circ.x([qubits[0]])
            circ.measure(qubits[-1], clbits[-1])
            circ.barrier()
            with circ.if_test((clbits[-1], 1)):
                circ.x(qubits[0])
                circ.x(qubits[-1])
        elif ff_type == "X_c1":
            circ.x(qubits)
            circ.barrier()
            circ.measure(qubits[-1], clbits[-1])
            with circ.if_test((clbits[-1], 1)):
                circ.x(qubits)
        elif ff_type == "X_c0":
            circ.measure(qubits[-1], clbits[-1])
            with circ.if_test((clbits[-1], 1)):
                circ.x(qubits)
        elif ff_type == "Z_c1":
            circ.z(qubits[:-1])
            circ.x(qubits[-1])
            circ.barrier()
            circ.measure(qubits[-1], clbits[-1])
            with circ.if_test((clbits[-1], 1)):
                circ.z(qubits[:-1])
                circ.x(qubits[-1])
        elif ff_type == "Z_c0":
            circ.measure(qubits[-1], clbits[-1])
            with circ.if_test((clbits[-1], 1)):
                circ.z(qubits[:-1])
                circ.x(qubits[-1])
        elif ff_type == "I_c0":
            circ.measure(qubits[-1], clbits[-1])
            circ.barrier()
            # uses repeated Z instead of identity to make sure it uses the same feedforward timing
            with circ.if_test((clbits[-1], 1)):
                circ.x(qubits[-1])
                circ.z(qubits[:-1])
                circ.barrier()
                circ.z(qubits[:-1])
        elif ff_type == "I_c1":
            circ.x(qubits[-1])
            circ.barrier()
            circ.measure(qubits[-1], clbits[-1])
            # uses repeated Z instead of identity to make sure it uses the same feedforward timing
            with circ.if_test((clbits[-1], 1)):
                circ.x(qubits[-1])
                circ.z(qubits[:-1])
                circ.barrier()
                circ.z(qubits[:-1])
        elif ff_type == "Delay":
            meas_dt = durations.get("measure", self.physical_qubits[-1], "dt")
            circ.append(Delay(meas_dt, unit="dt"), [qubits[-1]], [])
            circ.barrier()
            circ.append(Delay(self.ff_delay, unit="dt"), [qubits[-1]], [])
        else:
            raise Exception(f"Not supporting {ff_type}")
        return circ

    def _generate_sequences(self, length: int, rng: Generator) -> Iterator[Clifford]:
        """Generate N+1 Clifford sequences with inverse at the end."""
        composed = Clifford([[1, 0], [0, 1]])
        for _ in range(length):
            elm = random_clifford(1, rng)
            composed = composed.compose(elm)
            yield elm
        if length > 0:
            yield composed.adjoint()

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


class DynamicCircuitsRBAnalysis(SubDecayFit):
    def __init__(
        self,
        physical_qubits,
        ff_operations,
        plot_measured_qubit=True,
        plot_summary=True,
    ):
        super().__init__()
        self.physical_qubits = physical_qubits
        self.ff_operations = ff_operations
        self.plot_summary = plot_summary
        self.plot_measured_qubit = plot_measured_qubit

    def _run_analysis(
        self,
        experiment_data,
    ):
        analysis_results, figs = [], []
        q_m = self.physical_qubits[-1]
        for ff_type in self.ff_operations:
            for i, q in enumerate(self.physical_qubits):
                name = f"{ff_type}(Q{q}_M{q_m})"
                self.set_options(
                    data_processor=DataProcessor(
                        "counts", [MarginalizeCounts({i}), Probability("0")]
                    ),
                    result_parameters=[curve.ParameterRepr("alpha", name)],
                    filter_data={"ff_type": ff_type},
                )
                self._name = name
                self.plotter.set_figure_options(
                    xlabel="Number of FF operation",
                    ylabel="P(0)",
                    figure_title=f"Data qubit: {q}, Measured qubit: {q_m}  Operation: {ff_type}",
                )
                analysis_result, fig = super()._run_analysis(experiment_data)
                analysis_results += analysis_result
                if q == q_m and not self.plot_measured_qubit:
                    continue
                figs += fig

        if self.plot_summary:
            results_fig = Figure(figsize=(6, 4))
            results_separate_fig = Figure(figsize=(len(self.physical_qubits) * 1.4, 4))
            _ = FigureCanvasSVG(results_fig)
            _ = FigureCanvasSVG(results_separate_fig)
            ax = results_fig.subplots(1, 1)
            axs = results_separate_fig.subplots(1, len(self.physical_qubits))
            x = np.arange(len(self.physical_qubits))
            x_ticks = [f"Q{q}" for q in self.physical_qubits]
            for ff_type in self.ff_operations:
                ys, y_errs = [], []
                for i, q in enumerate(self.physical_qubits):
                    alpha = next(
                        filter(
                            lambda res: res.name == f"{ff_type}(Q{q}_M{q_m})",
                            analysis_results,
                        )
                    )
                    y, y_err = alpha.value.n, alpha.value.s
                    ys.append(y)
                    y_errs.append(y_err)
                    axs[i].errorbar(
                        [1],
                        y,
                        yerr=y_err,
                        fmt="o",
                        alpha=0.5,
                        capsize=4,
                        markersize=5,
                        label=ff_type,
                    )
                ax.errorbar(
                    x,
                    ys,
                    yerr=y_errs,
                    fmt="o",
                    alpha=0.5,
                    capsize=4,
                    markersize=5,
                    label=ff_type,
                )
            ax.legend()
            ax.set_xticks(x, x_ticks)
            ax.set_title(f"Measured qubit: {q_m}")
            for i, q in enumerate(self.physical_qubits):
                axs[i].set_xticks([1], [f"Q{q}"])
            axs[-1].set_xticks([1], [f"Q{q_m}:M"])
            axs[-1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
            results_separate_fig.tight_layout()
            figs += [results_fig, results_separate_fig]

        return analysis_results, figs

    @classmethod
    def _default_options(cls):
        default_options = super()._default_options()
        default_options.plot_raw_data = True
        default_options.average_method = "sample"

        return default_options
