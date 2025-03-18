# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.models.exceptions import BackendPropertyError
from qiskit.transpiler import AnalysisPass, CouplingMap, PassManager
from qiskit.transpiler.passes import VF2Layout
from qiskit.transpiler.passes.layout.vf2_utils import ErrorMap
from qiskit_ibm_runtime import IBMBackend


def build_error_dataframe(backend: IBMBackend) -> pd.DataFrame:
    data = []
    props = backend.properties()
    gate_name_2q = list({"ecr", "cx", "cz"}.intersection(backend.basis_gates))[0]

    for q in range(backend.num_qubits):
        data.append(
            {
                "metric": "faulty",
                "value": np.nan if not props.is_qubit_operational(q) else 0,
                "qubits": (q, q),
                "sign": +1,
            }
        )

    for q in range(backend.num_qubits):
        try:
            t1 = props.t1(q)
        except BackendPropertyError:
            t1 = np.nan
        data.append(
            {
                "metric": "t1",
                "value": t1,
                "qubits": (q, q),
                "sign": -1,
            }
        )

    for q in range(backend.num_qubits):
        try:
            t2 = props.t2(q)
        except BackendPropertyError:
            t2 = np.nan
        data.append(
            {
                "metric": "t2",
                "value": t2,
                "qubits": (q, q),
                "sign": -1,
            }
        )

    for q in range(backend.num_qubits):
        try:
            ro_err = props.readout_error(q)
        except BackendPropertyError:
            ro_err = np.nan
        data.append(
            {
                "metric": "readout_error",
                "value": ro_err,
                "qubits": (q, q),
                "sign": +1,
            }
        )

    for edge in list(backend.coupling_map):
        try:
            gate_err_2q = props.gate_error(gate_name_2q, edge)
            if gate_err_2q == 1.0:
                gate_err_2q = np.nan
        except BackendPropertyError:
            gate_err_2q = np.nan
        data.append(
            {
                "metric": "gate_err_2q",
                "value": gate_err_2q,
                "qubits": edge,
                "sign": +1,
            }
        )

    return pd.DataFrame(data)


def compute_error_dataframe(
    df: pd.DataFrame, weights: dict[str, float]
) -> pd.DataFrame:
    df = df.copy()

    if set(df["metric"]) != set(weights.keys()):
        missing_keys = set(df["metric"]) - set(weights.keys())
        raise ValueError(f"Missing weights for: {missing_keys}")

    # Drop any qubits which have missing properties
    bad_edges = list(df[df["value"].isna()]["qubits"])
    bad_qubits = set([item for sublist in bad_edges for item in sublist])
    df = df[
        df["qubits"].map(
            lambda q: (q[0] not in bad_qubits) and (q[1] not in bad_qubits)
        )
    ]

    # Sign indicates whether a metric should be small or large
    df["value"] *= df["sign"]
    df.drop(columns=["sign"], inplace=True)

    # Normalize values about the means
    df_mean = df.groupby(["metric"]).agg({"value": "mean"})
    df = df.set_index(["metric", "qubits"]) - df_mean
    df_std = df.groupby(["metric"]).agg({"value": "std"})
    df_std[df_std["value"] == 0.0] = 1.0
    df = df / df_std

    # Apply weights for properties
    df_weights = pd.DataFrame([{"metric": m, "value": v} for m, v in weights.items()])
    df_weights = df_weights.set_index("metric")
    df = df * df_weights
    df.reset_index(inplace=True)

    ## Aggregate over metrics
    df = df.groupby(["qubits"]).agg({"value": "mean"}).reset_index()
    df["value"] -= df["value"].min()

    return df


def build_error_map(backend, weights_dict, symmetrize: bool = True):
    df_props = build_error_dataframe(backend)
    df_err = compute_error_dataframe(df_props, weights_dict)

    HI_ERR_CONST = 1e3

    err_dict = {}
    for edge in list(backend.coupling_map):
        err_dict[edge] = HI_ERR_CONST
        err_dict[tuple(reversed(edge))] = HI_ERR_CONST
    for q in range(backend.num_qubits):
        err_dict[(q, q)] = HI_ERR_CONST

    for _, (qubits, err_rate) in df_err.iterrows():
        err_dict[qubits] = err_rate
        err_dict[tuple(reversed(qubits))] = err_rate

    error_map = ErrorMap(len(err_dict))
    for edge, err_rate in err_dict.items():
        error_map.add_error(edge, err_rate)

    return df_err, error_map


class VF2WeightedLayout(AnalysisPass):
    def __init__(self, weights_dict: dict[str, float], backend: IBMBackend):
        super().__init__()
        self._backend = backend
        self._weights_dict = weights_dict

    def run(self, dag):
        df, error_map = build_error_map(self._backend, self._weights_dict)
        self.property_set["vf2_avg_error_map"] = error_map
        self.property_set["vf2_dataframe"] = df
        return dag


def dummy_path_circuit(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    for i in range(0, num_qubits - 1, 2):
        qc.cz(i, i + 1)
    for i in range(1, num_qubits - 1, 2):
        qc.cz(i, i + 1)

    return qc


def symmetrize_coupling_map(cm: CouplingMap) -> CouplingMap:
    edge_list = set()
    for edge in list(cm):
        edge_list |= {tuple(edge)}
        edge_list |= {tuple(reversed(edge))}
    return CouplingMap(edge_list)


def get_optimal_path(
    weights_dict: dict[str, float],
    backend: IBMBackend,
    num_qubits: int,
    seed: int = 42,
    time_limit: float = 30.0,
    max_trials: int = -1,
    call_limit=None,
):
    pm = PassManager(
        [
            VF2WeightedLayout(weights_dict=weights_dict, backend=backend),
            VF2Layout(
                strict_direction=False,
                seed=seed,
                coupling_map=symmetrize_coupling_map(backend.coupling_map),
                time_limit=time_limit,
                max_trials=max_trials,
                call_limit=call_limit,
            ),
        ]
    )
    pm.run(dummy_path_circuit(num_qubits))
    qubit_mapping = {
        k._index: v for k, v in pm.property_set["layout"].get_virtual_bits().items()
    }
    mapped_path = [qubit_mapping[i] for i in range(num_qubits)]
    return tuple(mapped_path)
