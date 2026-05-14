# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Utilities for qubit and layout characterization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Literal
import copy
import numpy as np
import matplotlib.pyplot as plt
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from pandas import DataFrame

from qiskit import QuantumCircuit
from qiskit.result import marginal_counts as mcts

from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.models import BackendProperties

from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
from qiskit_experiments.library import StandardRB
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity
from qiskit_experiments.library import T1, T2Hahn

import qiskit_device_benchmarking.utilities.graph_utils as gu
import qiskit_device_benchmarking.utilities.layer_fidelity_utils as lfu


# Experiment type constants
EXPERIMENT_READOUT = "readout"
EXPERIMENT_RB_1Q = "rb_1q"
EXPERIMENT_RB_2Q = "rb_2q"
EXPERIMENT_T1 = "t1"
EXPERIMENT_T2 = "t2"

# Type hint for supported experiments
SupportedExperiments = Literal["readout", "rb_1q", "rb_2q", "t1", "t2"]

logger = logging.getLogger(__name__)

class BackendCharacterization:
    """Class for backend characterization."""

    def __init__(self, backend: IBMBackend):
        """Initialize the backend characterization class."""
        self._backend = backend
        self._jobs = {}
        self._experiments = {}
        self._new_data = {}

    @property
    def backend(self) -> IBMBackend:
        """Return the backend."""
        return self._backend

    @property
    def allowed_experiments(self) -> List[SupportedExperiments]:
        """Return the list of allowed experiments."""
        return [EXPERIMENT_READOUT, EXPERIMENT_RB_1Q, EXPERIMENT_RB_2Q, EXPERIMENT_T1, EXPERIMENT_T2]

    @property
    def jobs(self) -> dict:
        """Return a dictionary of jobs. Keys are experiment names, values are lists of jobs."""
        return self._jobs

    def run_and_update(
        self,
        experiments: Optional[Iterable[SupportedExperiments]] = None,
        shots: Optional[Dict[str, int]] = None,
    ) -> IBMBackend:
        """Run selected characterization experiments on the given IBM Quantum backend and update its properties.

        Default configuration for each experiment:
        - RB(1q): lengths=[1,50,100,500,1000,3000], num_samples=6, seed=42,
                    separate_jobs=True, max_circuits=3 * num_samples
        - LF(2q): lengths=[1,10,20,30,40,60,80,100,150,200,400], num_samples=12,
                    seed=60, max_circuits=144
        - T1/T2 delays: [1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6]
        - Two-qubit gate preference: ecr -> cz -> cx

        Args:
            experiments: One or more of {"readout", "rb_1q", "rb_2q", "t1", "t2"}.

            * "readout": spam0/spam1 over all qubits
            * "rb_1q": StandardRB per qubit, batched via Parallel+BatchExperiment
            * "rb_2q": LayerFidelity on horizontal & vertical grid layers
            * "t1": T1 on all qubits in parallel
            * "t2": T2Hahn on all qubits in parallel

            If None, runs all supported experiments.

            shots: Optional dict overriding shots, keys in {"readout","rb_1q","rb_2q","t1","t2"}.
                If None, default shots are used for each experiment:
                readout=10000, rb_1q=250, rb_2q=250, t1=250, t2=250
        Raises:
            ValueError: if an unknown experiment is requested.
        """
        self.run_experiments(
            experiments=experiments,
            shots=shots
        )
        self.analyze_results()
        return self.update_backend()

    def run_experiments(
        self,
        experiments: Optional[Iterable[SupportedExperiments]] = None,
        shots: Optional[Dict[str, int]] = None,
    ) -> dict:
        """Run selected characterization experiments.

        Default configuration for each experiment:
        - RB(1q): lengths=[1,50,100,500,1000,3000], num_samples=6, seed=42,
                    separate_jobs=True, max_circuits=3 * num_samples
        - LF(2q): lengths=[1,10,20,30,40,60,80,100,150,200,400], num_samples=12,
                    seed=60, max_circuits=144
        - T1/T2 delays: [1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6]
        - Two-qubit gate preference: ecr -> cz -> cx

        Args:
            experiments: One or more of {"readout", "rb_1q", "rb_2q", "t1", "t2"}.

            * "readout": spam0/spam1 over all qubits
            * "rb_1q": StandardRB per qubit, batched via Parallel+BatchExperiment
            * "rb_2q": LayerFidelity on horizontal & vertical grid layers
            * "t1": T1 on all qubits in parallel
            * "t2": T2Hahn on all qubits in parallel

            If None, runs all supported experiments.

            shots: Optional dict overriding shots, keys in {"readout","rb_1q","rb_2q","t1","t2"}.
                If None, default shots are used for each experiment:
                readout=10000, rb_1q=250, rb_2q=250, t1=250, t2=250

        Raises:
            ValueError: if an unknown experiment is requested.

        Returns:
            A dictionary of experiment type to job mapping.
        """
        # ---- Local configuration  ----
        allowed_experiments = {"readout", "rb_1q", "rb_2q", "t1", "t2"}

        rb1q_lengths = np.array([1, 50, 100, 500, 1000, 3000])
        rb1q_num_samples = 6
        rb1q_seed = 42
        rb1q_samples_m = 3  # => max_circuits = 18

        lf_lengths = [1, 10, 20, 30, 40, 60, 80, 100, 150, 200, 400]
        lf_num_samples = 12
        lf_seed = 60
        lf_max_circuits = 144

        t1_delays = [1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6]
        t2_delays = [1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6]

        default_shots = {EXPERIMENT_READOUT: 10_000, EXPERIMENT_RB_1Q: 250, EXPERIMENT_RB_2Q: 250, EXPERIMENT_T1: 300, EXPERIMENT_T2: 300}
        shots = {**default_shots, **(shots or {})}

        chosen = set(experiments) if experiments is not None else allowed_experiments
        unknown = chosen - allowed_experiments
        if unknown:
            raise ValueError(f"Unsupported experiments: {sorted(unknown)}. Allowed: {allowed_experiments}")

        # ---- Build characterization experiments ----
        sampler = Sampler(mode=self.backend)

        readout_circuits: Optional[List[QuantumCircuit]] = None
        if EXPERIMENT_READOUT in chosen:
            logger.info("Building readout experiments")
            readout_circuits = self._build_readout_circuits()

        sqrb_exp: Optional[BatchExperiment] = None
        if EXPERIMENT_RB_1Q in chosen:
            logger.info("Building 1Q RB experiments")
            sqrb_exp = self._build_oneq_rb_experiments(
                lengths=rb1q_lengths,
                num_samples=rb1q_num_samples,
                seed=rb1q_seed,
                samples_m=rb1q_samples_m,
            )

        lf_h = lf_v = None
        layers: Optional[List[List[Tuple[int, int]]]] = None
        if EXPERIMENT_RB_2Q in chosen:
            logger.info("Building 2Q RB experiments")
            lf_h, lf_v, layers = self._build_layered_rb_experiments(
                lengths=lf_lengths,
                num_samples=lf_num_samples,
                seed=lf_seed,
                max_circuits=lf_max_circuits,
            )

        t1_exp: Optional[ParallelExperiment] = None
        if EXPERIMENT_T1 in chosen:
            logger.info("Building T1 experiments")
            t1_exp = self._build_t1_experiments(delays=t1_delays)

        t2_exp: Optional[ParallelExperiment] = None
        if EXPERIMENT_T2 in chosen:
            logger.info("Building T2 experiments")
            t2_exp = self._build_t2_experiments(delays=t2_delays)

        # ---- Run and collect results ----
        if lf_h and lf_v:
            sampler.options.environment.job_tags = ["characterization", "rb_2q"]
            sampler.options.default_shots = int(shots[EXPERIMENT_RB_2Q])
            lf_v_exp_data = lf_v.run(sampler=sampler)
            lf_h_exp_data = lf_h.run(sampler=sampler)
            self._jobs[EXPERIMENT_RB_2Q] = lf_v_exp_data.jobs() + lf_h_exp_data.jobs()
            self._experiments[EXPERIMENT_RB_2Q] = {"lf_v": lf_v_exp_data, "lf_h": lf_h_exp_data, "layers": layers}
            logger.info("Layered two-qubit RB submitted: %s", lf_v_exp_data.job_ids + lf_h_exp_data.job_ids)

        if readout_circuits:
            sampler.options.environment.job_tags = ["characterization", "readout"]
            job_readout = sampler.run(readout_circuits, shots=int(shots[EXPERIMENT_READOUT]))
            self._jobs[EXPERIMENT_READOUT] = [job_readout]
            self._experiments[EXPERIMENT_READOUT] = job_readout
            logger.info("Readout job submitted: %s", job_readout.job_id())

        if t1_exp:
            sampler.options.environment.job_tags = ["characterization", "t1"]
            sampler.options.default_shots = int(shots[EXPERIMENT_T1])
            t1_exp_data = t1_exp.run(sampler=sampler)
            self._jobs[EXPERIMENT_T1] = t1_exp_data.jobs()
            self._experiments[EXPERIMENT_T1] = t1_exp_data
            logger.info("T1 experiment submitted: %s", t1_exp_data.job_ids)

        if t2_exp:
            sampler.options.environment.job_tags = ["characterization", "t2"]
            sampler.options.default_shots = int(shots[EXPERIMENT_T2])
            job_t2 = t2_exp.run(sampler=sampler)
            self._jobs[EXPERIMENT_T2] = job_t2.jobs()
            self._experiments[EXPERIMENT_T2] = job_t2
            logger.info("T2 (Hahn) experiment submitted: %s", job_t2.job_ids)

        if sqrb_exp:
            sampler.options.environment.job_tags = ["characterization", "rb_1q"]
            sampler.options.default_shots = int(shots[EXPERIMENT_RB_1Q])
            sqrb_exp_data = sqrb_exp.run(sampler=sampler)
            self._jobs[EXPERIMENT_RB_1Q] = sqrb_exp_data.jobs()
            self._experiments[EXPERIMENT_RB_1Q] = sqrb_exp_data
            logger.info("Single-qubit RB submitted: %s", sqrb_exp_data.job_ids)

        return self._jobs

    def analyze_results(self) -> dict:
        """Extract error rates and other properties from the experiment results.

        Returns:
            A dictionary of different error maps.
        """
        # ---- Build error / property maps ----
        if EXPERIMENT_READOUT in self._experiments:
            job_readout = self._jobs[EXPERIMENT_READOUT][0]
            readout_result = job_readout.result()
            self._new_data["readout_error"] = self._get_readout_errors(readout_result)

        if EXPERIMENT_RB_1Q in self._experiments:
            sqrb_exp_data = self._experiments[EXPERIMENT_RB_1Q]
            sqrb_result_x = sqrb_exp_data.analysis_results("EPG_x", dataframe=True)
            self._new_data["oneq_error_x"] = self._get_oneq_properties(sqrb_result_x)
            sqrb_result_sx = sqrb_exp_data.analysis_results("EPG_sx", dataframe=True)
            self._new_data["oneq_error_sx"]  = self._get_oneq_properties(sqrb_result_sx)

        if EXPERIMENT_RB_2Q in self._experiments:
            lf_v_exp_data = self._experiments[EXPERIMENT_RB_2Q]["lf_v"]
            lf_h_exp_data = self._experiments[EXPERIMENT_RB_2Q]["lf_h"]
            layers = self._experiments[EXPERIMENT_RB_2Q]["layers"]
            lf_result_v = lf_v_exp_data.analysis_results("ProcessFidelity", dataframe=True)
            lf_result_h = lf_h_exp_data.analysis_results("ProcessFidelity", dataframe=True)
            self._new_data["lf_error_map"]  = self._get_twoq_errors(lf_result_h, lf_result_v, layers)

        if EXPERIMENT_T1 in self._experiments:
            t1_exp_data = self._experiments[EXPERIMENT_T1]
            t1_df = t1_exp_data.analysis_results(dataframe=True)
            self._new_data["t1_map"]= self._get_oneq_properties(t1_df)

        if EXPERIMENT_T2 in self._experiments:
            t2_exp_data = self._experiments[EXPERIMENT_T2]
            t2_df = t2_exp_data.analysis_results(dataframe=True)
            self._new_data["t2_map"] = self._get_oneq_properties(t2_df)

        return self._new_data


    def update_backend(self) -> IBMBackend:
        """Update backend with the experiment results.

        Returns:
            IBMBackend: _description_
        """
        # ---- Update backend properties ----
        backend = copy.deepcopy(self.backend)
        props_dict = backend.properties().to_dict()

        if "readout_error" in self._new_data:
            logger.info("Updating readout error")
            props_dict = self._update_qubit_props(
                props_dict, "readout_error", self._new_data["readout_error"])

        if "oneq_error_x" in self._new_data:
            logger.info("Updating single-qubit X error")
            props_dict = self._update_1q_errors(
                props_dict, self._new_data["oneq_error_x"], prop_name='x')

        if "oneq_error_sx" in self._new_data:
            logger.info("Updating single-qubit SX error")
            props_dict = self._update_1q_errors(
                props_dict, self._new_data["oneq_error_sx"], prop_name='sx')

        if "lf_error_map" in self._new_data:
            logger.info("Updating two-qubit error")
            props_dict = self._update_lf_errors(
                props_dict, self._new_data["lf_error_map"])

        if "t1_map" in self._new_data:
            logger.info("Updating T1")
            props_dict = self._update_qubit_props(
                props_dict, "T1", self._new_data["t1_map"])

        if "t2_map" in self._new_data:
            logger.info("Updating T2")
            props_dict = self._update_qubit_props(
                props_dict, "T2", self._new_data["t2_map"])

        props = BackendProperties.from_dict(props_dict)
        backend._properties = props  # intentional: preserved behavior

        return backend

    def _build_readout_circuits(self) -> List[QuantumCircuit]:
        """Create a SPAM experiment on all qubits on parallel."""
        num_qubits = self.backend.num_qubits
        spam0 = QuantumCircuit(num_qubits, num_qubits)
        spam0.measure_all()
        spam1 = QuantumCircuit(num_qubits, num_qubits)
        spam1.x(range(num_qubits))
        spam1.measure_all()
        return [spam0, spam1]

    def _build_oneq_rb_experiments(
        self,
        lengths: np.ndarray,
        num_samples: int,
        seed: int,
        samples_m: int,
    ) -> BatchExperiment:
        """Standard 1Q RB across independent qubit sets."""
        undirected_graph: Any = self.backend.coupling_map.graph.to_undirected(multigraph=False)
        sqrb_batches = gu.get_iso_qubit_list(undirected_graph)

        sqrb_exp_list: List[ParallelExperiment] = []
        for batch in sqrb_batches:
            rb1q_exps = []
            for q in batch:
                rb1q_exps.append(
                    StandardRB(
                        physical_qubits=[int(q)],
                        lengths=lengths,
                        backend=self.backend,
                        seed=seed,
                        num_samples=num_samples,
                    )
                )
            sqrb_exp_list.append(ParallelExperiment(rb1q_exps, backend=self.backend, flatten_results=True))

        sqrb_exp = BatchExperiment(sqrb_exp_list, backend=self.backend, flatten_results=True)
        sqrb_exp.set_experiment_options(separate_jobs=True)
        sqrb_exp.experiment_options.max_circuits = samples_m * num_samples
        return sqrb_exp

    def _build_layered_rb_experiments(
        self,
        lengths: List[int],
        num_samples: int,
        seed: int,
        max_circuits: int,
    ) -> Tuple[LayerFidelity, LayerFidelity, List[List[Tuple[int, int]]]]:
        """Build LayerFidelity experiments for horizontal and vertical chains and grids.

        Returns:
            (lf_h, lf_v, layers)
        """
        twoq_gate = self._get_twoq_gate()
        oneq_gates = self._get_oneq_basis()

        grid_chains = lfu.get_grids(self.backend)
        coupling_map = self.backend.coupling_map
        edges = list(self.backend.target[twoq_gate].keys())

        layers: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
        grid_chain_flt = [[], []]

        for i in range(2):  # 0=H, 1=V
            all_pairs = gu.path_to_edges(grid_chains[i], coupling_map)
            for j, pair_lst in enumerate(all_pairs):
                grid_chain_flt[i] += grid_chains[i][j]
                sub_pairs = [tuple(p) if tuple(p) in edges else tuple(p)[::-1] for p in pair_lst]
                layers[2 * i] += sub_pairs[0::2]
                layers[2 * i + 1] += sub_pairs[1::2]

        h_qubits = grid_chain_flt[0]
        v_qubits = grid_chain_flt[1]

        lf_h = LayerFidelity(
            physical_qubits=h_qubits,
            two_qubit_layers=[layers[0], layers[1]],
            lengths=lengths,
            backend=self.backend,
            num_samples=num_samples,
            seed=seed,
            two_qubit_gate=twoq_gate,
            one_qubit_basis_gates=oneq_gates,
        )
        lf_v = LayerFidelity(
            physical_qubits=v_qubits,
            two_qubit_layers=[layers[2], layers[3]],
            lengths=lengths,
            backend=self.backend,
            num_samples=num_samples,
            seed=seed,
            two_qubit_gate=twoq_gate,
            one_qubit_basis_gates=oneq_gates,
        )

        lf_h.experiment_options.max_circuits = max_circuits
        lf_v.experiment_options.max_circuits = max_circuits

        return lf_h, lf_v, layers

    def _build_t1_experiments(
        self,
        delays: List[float],
    ) -> ParallelExperiment:
        """Create T1 experiments on all qubits in parallel."""
        qubits = list(range(self.backend.num_qubits))
        t1_exp = ParallelExperiment(
            [
                T1(
                    physical_qubits=[q],
                    delays=delays,
                )
                for q in qubits
            ],
            backend=self.backend,
            analysis=None,
            flatten_results=True,
        )
        return t1_exp

    def _build_t2_experiments(
        self,
        delays: List[float],
    ) -> ParallelExperiment:
        """Create T2-Hahn experiments on all qubits in parallel."""
        qubits = list(range(self.backend.num_qubits))
        t2_exp = ParallelExperiment(
            [
                T2Hahn(
                    physical_qubits=[q],
                    delays=delays,
                )
                for q in qubits
            ],
            backend=self.backend,
            analysis=None,
            flatten_results=True,
        )
        return t2_exp

    def _get_twoq_gate(self) -> str:
        """Get native 2Q gate of a device"""
        basis = self.backend.configuration().basis_gates
        if "ecr" in basis:
            return "ecr"
        if "cz" in basis:
            return "cz"
        return "cx"

    def _get_oneq_basis(self) -> List[str]:
        """Get one-qubit basis gates excluding the chosen two-qubit gate; skip rx/rzz."""
        oneq = []
        twoq_gate = self._get_twoq_gate()
        for g in self.backend.configuration().basis_gates:
            if g.casefold() in {"rx", "rzz", "xslow"}:
                continue
            if g.casefold() != twoq_gate.casefold():
                oneq.append(g)
        return oneq

    def _get_readout_errors(self, readout_result) -> Dict[int, float]:
        """Get readout errors per qubit from a job result."""
        num_qubits = self.backend.num_qubits
        ro_error: Dict[int, float] = {}
        cts_spam0 = readout_result[0].data.meas.get_counts()
        cts_spam1 = readout_result[1].data.meas.get_counts()
        spam_shots = readout_result[0].data.meas.num_shots
        for q in range(num_qubits):
            try:
                ro_error[q] = 1 - ((mcts(cts_spam0, [q])["0"] + mcts(cts_spam1, [q])["1"]) / 2) / spam_shots
            except KeyError:
                ro_error[q] = 1 - (mcts(cts_spam0, [q])["0"] / 2) / spam_shots
        return ro_error

    def _get_oneq_properties(self, job_results_df: DataFrame) -> Dict[Union[int, Tuple[int, int]], float]:
        """Extract 1Q properties (1Q error, T1, T2) from a job result and return a
        corresponding dictionary."""
        values_map: Dict[Union[int, Tuple[int, int]], float] = {}
        for _, row in job_results_df.iterrows():
            key = row.components[0].index
            values_map[key] = row.value.nominal_value
        return values_map

    def _get_twoq_errors(
        self,
        lf_result_h_df: DataFrame,
        lf_result_v_df: DataFrame,
        layers: List[List[Tuple[int, int]]],
    ) -> Dict[str, float]:
        """Extract 2Q errors from layer fidelity experiments (horiztonal and vertical) and return a
        corresponding error dictionary."""
        twoq_gate = self._get_twoq_gate()
        lf_err_dict = lfu.make_error_dict(self.backend, twoq_gate)

        updated_err_dicts = []
        for i, lf_df in enumerate([lf_result_h_df, lf_result_v_df]):
            for j in range(2):
                updated_err_dicts.append(lfu.df_to_error_dict(lf_df, layers[2 * i + j]))

        lf_err_dict = lfu.update_error_dict(lf_err_dict, updated_err_dicts)
        return lf_err_dict

    def _update_qubit_props(self, props: Dict, prop_name: str, values_map: Dict[int, float]) -> Dict:
        """
        Update a per-qubit property in props["qubits"].

        Examples:
            props = _update_qubit_props(props, "readout_error", ro_error_map)
            props = _update_qubit_props(props, "T1", t1_map)
            props = _update_qubit_props(props, "T2", t2_map)
        """
        for q, val in values_map.items():
            for param in props["qubits"][q]:
                if param["name"] == prop_name:
                    if prop_name in ["T1", "T2"]:
                        param["value"] = float(val) * 10**6 # convert to micro seconds unit
                    else:
                        param["value"] = float(val)
                    break
        return props

    def _update_1q_errors(self, props: Dict, error_map: Dict[Union[int, Tuple[int, int]], float], prop_name='x') -> Dict:
        """Update 1q gate_error for sx/x from EPG maps."""
        for ix, gate in enumerate(props["gates"]):
            gate_type, component = gate['gate'], int(gate['qubits'][0])

            if gate_type == prop_name:
                for iy, parameter in enumerate(gate["parameters"]):
                    if parameter["name"] == "gate_error" and component in error_map:
                        gate_error = error_map[component]
                        props["gates"][ix]["parameters"][iy]["value"] = gate_error
                        break
        return props


    def _update_lf_errors(self, props: Dict, error_map: Dict[str, float]) -> Dict:
        """Update 2q gate_error for {cz,cx,ecr} from LF pair->error map."""
        for ix, gate in enumerate(props["gates"]):
            if gate["gate"] in ["cz", "cx", "ecr"]:
                q0, q1 = gate["qubits"]
                pair_key = f"{q0}_{q1}"
                pair_key_rev = f"{q1}_{q0}"
                if pair_key in error_map or pair_key_rev in error_map:
                    gate_error = error_map.get(pair_key, error_map.get(pair_key_rev))
                    for iy, param in enumerate(gate["parameters"]):
                        if param["name"] == "gate_error":
                            props["gates"][ix]["parameters"][iy]["value"] = float(gate_error)
                            break
        return props


# ----------------------------- Plotting Functions --------------------------------- #

def plot_characterization_comparison(
    old_props: Dict,
    new_props: Dict,
    plots: Iterable[str],
    title_prefix=None,
    log_scale: Optional[Dict[str, bool]] = None,
):
    """
    Plot measured vs reported comparisons for selected characterization results.

    plots:
      "readout", "rb_1q_sx", "rb_1q_x", "rb_2q", "t1", "t2"
    """
    prefix = (title_prefix + " – ") if title_prefix else ""

    use_log = {
        "readout": False,
        "rb_1q_sx": True,
        "rb_1q_x": True,
        "rb_2q": True,
        "t1": True,
        "t2": True,
    }
    if log_scale:
        use_log.update(log_scale)

    plot_configs = {
        "readout": {
            "extract": lambda props: _extract_qubit_property(props, "readout_error"),
            "title": "Readout Error — measured vs reported",
            "xlabel": "Qubit",
            "ylabel": "Readout error",
            "figsize": (18, 5.5),
            "labels": lambda keys: [str(k) for k in keys],
        },
        "rb_1q_sx": {
            "extract": _extract_1q_sx_errors,
            "title": "1Q RB EPG (sx) — measured vs reported",
            "xlabel": "Qubit",
            "ylabel": "EPG (sx)",
            "figsize": (18, 5.5),
            "labels": lambda keys: [str(k) for k in keys],
        },
        "rb_1q_x": {
            "extract": _extract_1q_x_errors,
            "title": "1Q RB EPG (x) — measured vs reported",
            "xlabel": "Qubit",
            "ylabel": "EPG (x)",
            "figsize": (18, 5.5),
            "labels": lambda keys: [str(k) for k in keys],
        },
        "rb_2q": {
            "extract": _extract_twoq_gate_errors,
            "title": "Layered 2Q RB EPG — measured vs reported",
            "xlabel": "2Q gate",
            "ylabel": "EPG (2Q)",
            "figsize": (22, 6.5),
            "labels": lambda keys: [f"{i}-{j}" for (i, j) in keys],
        },
        "t1": {
            "extract": lambda props: _extract_qubit_property(props, "T1"),
            "title": "T1 — measured vs reported",
            "xlabel": "Qubit",
            "ylabel": "T1 (s)",
            "figsize": (18, 5.5),
            "labels": lambda keys: [str(k) for k in keys],
        },
        "t2": {
            "extract": lambda props: _extract_qubit_property(props, "T2"),
            "title": "T2 (Hahn) — measured vs reported",
            "xlabel": "Qubit",
            "ylabel": "T2 (s)",
            "figsize": (18, 5.5),
            "labels": lambda keys: [str(k) for k in keys],
        },
    }

    if 'rb_1q' in plots:
        plots.append('rb_1q_x')
        plots.append('rb_1q_sx')
        plots.remove('rb_1q')

    for name in plots:
        config = plot_configs[name]

        old_map = config["extract"](old_props)
        new_map = config["extract"](new_props)

        keys = _sorted_keys_by_new(old_map, new_map)
        if not keys:
            continue

        y_old = [old_map.get(k, np.nan) for k in keys]
        y_new = [new_map.get(k, np.nan) for k in keys]
        x = np.arange(len(keys))

        fig, ax = plt.subplots(figsize=config["figsize"])
        ax.set_title(f"{prefix}{config['title']} (sorted by measured)")
        _plot_lines(ax, x, y_old, y_new, lw=1.1, ms=3.0)

        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel(config["ylabel"])

        if use_log[name]:
            ax.set_yscale("log")

        _set_every_xtick_with_vertical_guides(
            ax,
            config["labels"](keys),
            rotation=90,
            fontsize=7,
        )

        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        plt.show()

def _extract_qubit_property(props_dict: Dict, prop_name: str) -> Dict[int, float]:
    """
    Generic extractor for per-qubit properties stored in props_dict["qubits"].

    Returns {qubit_index: value} for a given property name (e.g., "T1", "T2", "readout_error").
    """
    out = {}
    for q, q_params in enumerate(props_dict["qubits"]):
        for p in q_params:
            if p.get("name") == prop_name:
                out[q] = p.get("value")
                break
    return out


def _extract_gate_errors(
    props_dict: Dict,
    num_qubits: int,
    gate_names: List[str],
) -> Dict:
    """
    Extract gate_error values from props_dict["gates"].

    Returns:
        num_qubits == 1  -> {q: error}
        num_qubits == 2  -> {(q0, q1): error}
    """
    out = {}

    for g in props_dict["gates"]:
        if len(g["qubits"]) != num_qubits:
            continue

        gate = g["gate"]
        if gate not in gate_names:
            continue

        val = None
        for p in g["parameters"]:
            if p["name"] == "gate_error":
                val = p["value"]
                break
        if val is None:
            continue

        key = g["qubits"][0] if num_qubits == 1 else tuple(g["qubits"])
        out[key] = val

    return out


def _extract_1q_sx_errors(props_dict: Dict) -> Dict:
    return _extract_gate_errors(props_dict, num_qubits=1, gate_names=["sx"])


def _extract_1q_x_errors(props_dict: Dict) -> Dict:
    return _extract_gate_errors(props_dict, num_qubits=1, gate_names=["x"])


def _extract_twoq_gate_errors(props_dict: Dict) -> Dict:
    return _extract_gate_errors(props_dict, num_qubits=2, gate_names=["cx", "cz", "ecr"])


def _sorted_keys_by_new(old_map: Dict, new_map: Dict):
    """
    Return union of keys sorted by measured (new_map) value.
    NaNs are placed at the end.
    """
    keys = sorted(set(old_map.keys()) | set(new_map.keys()))
    return sorted(
        keys,
        key=lambda k: (np.isnan(new_map.get(k, np.nan)), new_map.get(k, np.nan)),
    )


def _set_every_xtick_with_vertical_guides(ax, labels, *, rotation=90, fontsize=7, weight="regular"):
    n = len(labels)
    idxs = np.arange(n)
    ax.set_xticks(idxs)
    ax.set_xticklabels(labels)

    for t in ax.get_xticklabels():
        t.set_rotation(rotation)
        t.set_fontsize(fontsize)
        t.set_fontweight(weight)
        if rotation == 90:
            t.set_horizontalalignment("center")
            t.set_verticalalignment("top")
        elif rotation == 45:
            t.set_horizontalalignment("right")
            t.set_verticalalignment("top")
        else:
            t.set_horizontalalignment("center")

    ax.tick_params(axis="x", which="both", width=0.6, length=3)

    for i in idxs:
        ax.axvline(i, color="k", alpha=0.08, linewidth=1.0, zorder=0)

    fig = ax.get_figure()
    fig.subplots_adjust(bottom=0.28 if rotation == 90 else 0.20)


def _plot_lines(ax, x_idx, y_old, y_new, *, lw=1.1, ms=3.0):
    ax.plot(x_idx, y_old, color="darkorange", marker="o", linewidth=lw, markersize=ms, label="Reported")
    ax.plot(x_idx, y_new, color="royalblue", marker="o", linewidth=lw, markersize=ms, label="Measured real-time")
