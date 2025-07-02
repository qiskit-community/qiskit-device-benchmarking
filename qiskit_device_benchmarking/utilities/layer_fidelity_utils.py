# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Utilities for layer fidelity
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import datetime

from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity
from qiskit.transpiler import CouplingMap
import qiskit_device_benchmarking.utilities.graph_utils as gu


def run_lf_chain(
    chain: list[int],
    backend: IBMBackend,
    nseeds: int = 6,
    seed: int = 42,
    cliff_lengths: list[int] = [1, 10, 20, 30, 40, 60, 80, 100, 150, 200, 400],
    nshots: int = 200,
) -> ExperimentData:
    """General function to run LF on a list of chains using default or
    command lie arguments. Note: If one hopes to run LF on a single chain,
    the single chain should be passed as a list with one element.
    Args:
    - best_qubit_chains: List of chains to run
    - backend
    - nseeds
    - cliff_lengths
    - nshots

    Returns:
    - List of ExperimentData objects
    """

    # Get two qubit gate
    if "ecr" in backend.configuration().basis_gates:
        twoq_gate = "ecr"
    elif "cz" in backend.configuration().basis_gates:
        twoq_gate = "cz"
    else:
        twoq_gate = "cx"

    # Get one qubit basis gates
    oneq_gates = []
    for i in backend.configuration().basis_gates:
        # put in a case to handle rx and rzz
        if i.casefold() == "rx" or i.casefold() == "rzz":
            continue
        if i.casefold() != twoq_gate.casefold():
            oneq_gates.append(i)

    # Get the coupling_map
    coupling_map = CouplingMap(backend.configuration().coupling_map)

    # Decompose chain into trivial two disjoint layers (list of list of gates)
    print("Decomposing qubit chain into two disjoint layers")
    all_pairs = gu.path_to_edges([chain], coupling_map)[0]
    all_pairs = [tuple(pair) for pair in all_pairs]  # make this is a list of tuples
    layers = [all_pairs[0::2], all_pairs[1::2]]  # will run a layer for each list

    # Check that each list is in the coupling map and is disjoint
    for layer in layers:
        for qpair in layer:
            if tuple(qpair) not in coupling_map:
                raise ValueError(f"Gate on {qpair} does not exist")

            for k in layer:
                if k == qpair:
                    continue

                if k[0] in qpair or k[1] in qpair:
                    print(f"Warning: sets are not disjoint for gate {k} and {qpair}")
    print("Disjoint layer 1")
    print(layers[0])
    print("Disjoint layer 2")
    print(layers[1])

    # Create LF instance
    lfexp = LayerFidelity(
        physical_qubits=chain,
        two_qubit_layers=layers,
        lengths=cliff_lengths,
        backend=backend,
        num_samples=nseeds,
        seed=seed,
        two_qubit_gate=twoq_gate,
        one_qubit_basis_gates=oneq_gates,
    )
    lfexp.set_experiment_options(max_circuits=2 * nseeds * len(cliff_lengths))

    # Generate all 2Q direct RB circuits
    circuits = lfexp.circuits()
    print(f"{len(circuits)} circuits are generated.")

    # Run LF experiment (generate circuits and submit the job)
    exp_data = lfexp.run(backend, shots=nshots)
    exp_data.auto_save = False
    print(
        f"Runing layer fidelity experiment: ID={exp_data.experiment_id} with jobs {exp_data.job_ids}]"
    )
    return exp_data


def get_rb_data(exp_data: ExperimentData) -> pd.DataFrame:
    """Extract raw and fitted RB data from a layer fidelity experiment.
    Data is exracted for all pairs of qubits in the chain.

    Args:
        exp_data: ExperimentData object

    Returns:
        rb_data_df: pd.DataFrame
    """
    rb_data_df = pd.DataFrame()
    df = exp_data.analysis_results(dataframe=True)
    if len(df) > 0:
        for exp_pair in exp_data.artifacts("curve_data"):
            curve_data_id = exp_pair.artifact_id
            # Retrieve qubits
            q1 = exp_data.artifacts(curve_data_id).device_components[0].index
            qubits = (q1,)
            if len(exp_data.artifacts(curve_data_id).device_components) > 1:
                q2 = exp_data.artifacts(curve_data_id).device_components[1].index
                qubits = (q1, q2)
            # Retrieve by ID
            scatter_table = exp_data.artifacts(curve_data_id).data
            data_df = scatter_table.dataframe
            data_df["qubits"] = [qubits] * len(data_df)
            rb_data_df = pd.concat([rb_data_df, data_df], ignore_index=True)
    return rb_data_df


def reconstruct_lf_per_length(
    exp_data: ExperimentData, qchain: list[int], backend: IBMBackend
) -> pd.DataFrame:
    """Extracts the lf and eplg values from a layer fidelity experiment for all
    subchain lengths ranging from l=4 to l= length of qchain. Saves the data in a
    DataFrame that contains the current legnth, the subchain with the highest lf value,
    and lf and eplg arrays for that given length.

    Args:
    - best_qubit_chains: List of chains to run
    - backend
    - nseeds
    - cliff_lengths
    - nshots

    Returns:
    - results_per_chain: pd.DataFrame
    """
    # Get the coupling_map
    coupling_map = CouplingMap(backend.configuration().coupling_map)

    # Recover layers from experiment
    all_pairs = gu.path_to_edges([qchain], coupling_map)[0]
    all_pairs = [tuple(pair) for pair in all_pairs]  # make sure these are all tuples
    layers = [all_pairs[0::2], all_pairs[1::2]]  # will run a layer for each list
    full_layer = [None] * (len(layers[0]) + len(layers[1]))
    full_layer[::2] = layers[0]
    full_layer[1::2] = layers[1]
    full_layer = [(qchain[0],)] + full_layer + [(qchain[-1],)]

    # Compute LF by chain length assuming the first layer is full with 2q-gates
    df = exp_data.analysis_results(dataframe=True)
    pfdf = df[df.name == "ProcessFidelity"]
    pfdf = pfdf.fillna({"value": 0})  # Fill Process Fidelity nan values with zeros
    results_per_chain = []
    # Check if the dataframe is empty
    if len(pfdf) > 0:
        pfs = [
            pfdf.loc[pfdf[pfdf.qubits == qubits].index[0], "value"]
            for qubits in full_layer
        ]
        pfs = list(map(lambda x: x.n if x != 0 else 0, pfs))
        pfs[0] = pfs[0] ** 2
        pfs[-1] = pfs[-1] ** 2
        pfs[0] = pfs[0] ** 2
        pfs[-1] = pfs[-1] ** 2

        # Approximate 1Q RB fidelities at both ends by the square root of 2Q RB fidelity at
        # both ends. For example, if we have [(0, 1), (1, 2), (2, 3), (3, 4)] 2Q RB fidelities
        # and if we want to compute a layer fidelity for [1, 2, 3], we approximate the 1Q
        # filedities for (1,) and (3,) by the square root of 2Q fidelities of (0, 1) and (3, 4).
        chain_lens = np.arange(4, len(pfs), 1)
        chain_fids = []
        for length in chain_lens:
            w = length + 1  # window size
            fid_w = max(
                np.sqrt(pfs[s])
                * np.prod(pfs[s + 1 : s + w - 1])
                * np.sqrt(pfs[s + w - 1])
                for s in range(len(pfs) - w + 1)
            )
            fid_w = []
            chains = []
            for s in range(len(pfs) - w + 1):
                fid_w.append(
                    np.sqrt(pfs[s])
                    * np.prod(pfs[s + 1 : s + w - 1])
                    * np.sqrt(pfs[s + w - 1])
                )
                chains.append(qchain[s : s + w - 1])

            chain = chains[
                np.argmax(fid_w)
            ]  # Get chain of length 'length' with highest fidelity
            chain_fids.append(np.max(fid_w))  # save highest fidelity
            lens = np.arange(4, length + 1, 1)

            # Get EPLG by chain length
            num_2q_gates = [L - 1 for L in lens]
            chain_eplgs = [
                4 / 5 * (1 - (fid ** (1 / num_2q)))
                for num_2q, fid in zip(num_2q_gates, chain_fids)
            ]
            results_per_chain.append(
                {
                    "qchain_length": len(chain),
                    "qchain": list(chain),
                    "lf": np.asarray(chain_fids),
                    "eplg": np.asarray(chain_eplgs),
                    "length": np.asarray(lens),
                    "job_ids": exp_data.job_ids,
                }
            )
    # Save data
    results_per_chain = pd.DataFrame.from_dict(results_per_chain)
    return results_per_chain


def make_lf_eplg_plots(
    backend: IBMBackend, exp_data: ExperimentData, chain: list[int], machine: str
):
    """Make layer fidelity and eplg plots for a given chain. Values
    are extracted from the provided experiment object.

    Args:
    - exp_data: ExperimentData object
    - chain: list of qubits in chain
    - machine: name of the backend
    """
    # Get the coupling_map
    coupling_map = CouplingMap(backend.configuration().coupling_map)

    # cover layers from experiment
    all_pairs = gu.path_to_edges([chain], coupling_map)[0]  # list of lists
    all_pairs = [tuple(pair) for pair in all_pairs]  # make sure these are all tuples
    layers = [all_pairs[0::2], all_pairs[1::2]]  # will run a layer for each list
    full_layer = [None] * (len(layers[0]) + len(layers[1]))
    full_layer[::2] = layers[0]
    full_layer[1::2] = layers[1]
    full_layer = [(chain[0],)] + full_layer + [(chain[-1],)]
    if len(full_layer) != len(chain) + 1:
        print(
            f"Length of full layer (= {len(full_layer)}) is not equal to length of the chain + 1 (= {len(chain) + 1})"
        )
        print("Make su the specified qubit chain is consistent with the device map")
        print(f"Full layer is {full_layer}")
        print(f"Chain is {chain}")
        exit()

    # Compute LF by chain length assuming the first layer is full with 2q-gates
    df = exp_data.analysis_results(dataframe=True)
    pfdf = df[df.name == "ProcessFidelity"]
    pfdf = pfdf.fillna({"value": 0})  # Fill Process Fidelity nan values with zeros
    # Check if the dataframe is empty
    if len(pfdf) > 0:
        pfs = [
            pfdf.loc[pfdf[pfdf.qubits == qubits].index[0], "value"]
            for qubits in full_layer
        ]
        pfs = list(map(lambda x: x.n if x != 0 else 0, pfs))
        pfs[0] = pfs[0] ** 2
        pfs[-1] = pfs[-1] ** 2
        pfs[0] = pfs[0] ** 2
        pfs[-1] = pfs[-1] ** 2

        # Approximate 1Q RB fidelities at both ends by the squa root of 2Q RB fidelity at
        # both ends. For example, if we have [(0, 1), (1, 2), (2, 3), (3, 4)] 2Q RB fidelities
        # and if we want to compute a layer fidelity for [1, 2, 3], we approximate the 1Q
        # filedities for (1,) and (3,) by the squa root of 2Q fidelities of (0, 1) and (3, 4).
        chain_lens = np.arange(4, len(pfs), 2)
        chain_fids = []
        for length in chain_lens:
            w = length + 1  # window size
            fid_w = max(
                np.sqrt(pfs[s])
                * np.prod(pfs[s + 1 : s + w - 1])
                * np.sqrt(pfs[s + w - 1])
                for s in range(len(pfs) - w + 1)
            )
            chain_fids.append(fid_w)

        # Get EPLG by chain length
        num_2q_gates = [length - 1 for length in chain_lens]
        chain_eplgs = [
            4 / 5 * (1 - (fid ** (1 / num_2q)))
            for num_2q, fid in zip(num_2q_gates, chain_fids)
        ]
    time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Plot LF by chain length
    fig1, ax1 = plt.subplots(figsize=(8.5, 4))
    title_lf = f"{machine} {time} LF for fixed {len(chain)}q chains"
    label_lf = f"Lf: {np.round(chain_fids[-1], 3)}"
    ax1.plot(chain_lens, chain_fids, marker="o", linestyle="-", label=label_lf)  # lfs
    ax1.set_title(title_lf, fontsize=11)
    ax1.set_xlim(0, chain_lens[-1] * 1.05)
    ax1.set_ylim(0.95 * min(chain_fids), 1)
    ax1.set_ylabel("Layer Fidelity (LF)")
    ax1.set_xlabel("Chain Length")
    ax1.grid()
    ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    ax1.figure.set_dpi(150)
    fig1.tight_layout()
    fig1.savefig(f"{machine}_lf.png")

    # Plot EPLG by chain length
    fig2, ax2 = plt.subplots(figsize=(8.5, 4))
    label_eplg = f"Eplg: {np.round(chain_eplgs[-1], 3)}"
    title_eplg = f"{machine} {time} EPLG for fixed {len(chain)}q chains"
    ax2.plot(
        chain_lens, chain_eplgs, marker="o", linestyle="-", label=label_eplg
    )  # eplgs
    ax2.set_title(title_eplg, fontsize=11)
    ax2.set_xlim(0, chain_lens[-1] * 1.05)
    ax2.set_ylabel("Error per Layed Gate (EPLG)")
    ax2.set_xlabel("Chain Length")
    ax2.grid()
    ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    ax2.figure.set_dpi(150)
    fig2.tight_layout()
    fig2.savefig(f"{machine}_eplg.png")


def get_lf_chain(backend: IBMBackend, chain_length: int = 100):
    """Get the chain reported from the backend for length chain_length

    Args:
    - backend: ibm backend
    - chain_length: the chain length to return

    Return:
    - chain: list of qubits of length chain_length (or None if not found)
    """

    for i in backend.properties().general_qlists:
        if i["name"] == "lf_%d" % chain_length:
            return i["qubits"]

    return None


def get_grids(
    backend: IBMBackend,
):
    """Return the grid chains for the different backends

    Args:
    - backend: ibm backend

    Return:
    - list of list of chains: list of list of chains for the grids to run. Each
    sub list is a list of chains that can be run in one experiment, each list in that sublist
    is a continuous path
    """

    grid_chains = []

    if backend.num_qubits == 127:
        horz_chain = [
            [
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                0,
                14,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                36,
                51,
                50,
                49,
                48,
                47,
                46,
                45,
                44,
                43,
                42,
                41,
                40,
                39,
                38,
                37,
                52,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                74,
                89,
                88,
                87,
                86,
                85,
                84,
                83,
                82,
                81,
                80,
                79,
                78,
                77,
                76,
                75,
                90,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                112,
                126,
                125,
                124,
                123,
                122,
                121,
                120,
                119,
                118,
                117,
                116,
                115,
                114,
                113,
            ]
        ]
        vert_chain = [
            [
                113,
                114,
                109,
                96,
                95,
                94,
                90,
                75,
                76,
                77,
                71,
                58,
                57,
                56,
                52,
                37,
                38,
                39,
                33,
                20,
                19,
                18,
                14,
                0,
                1,
                2,
                3,
                4,
                15,
                22,
                23,
                24,
                34,
                43,
                42,
                41,
                53,
                60,
                61,
                62,
                72,
                81,
                80,
                79,
                91,
                98,
                99,
                100,
                110,
                118,
                119,
                120,
                121,
                122,
                111,
                104,
                103,
                102,
                92,
                83,
                84,
                85,
                73,
                66,
                65,
                64,
                54,
                45,
                46,
                47,
                35,
                28,
                27,
                26,
                16,
                8,
                9,
                10,
                11,
                12,
                17,
                30,
                31,
                32,
                36,
                51,
                50,
                49,
                55,
                68,
                69,
                70,
                74,
                89,
                88,
                87,
                93,
                106,
                107,
                108,
                112,
                126,
            ]
        ]
    elif backend.num_qubits == 133:
        horz_chain = [
            [
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                0,
                15,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                37,
                52,
                51,
                50,
                49,
                48,
                47,
                46,
                45,
                44,
                43,
                42,
                41,
                40,
                39,
                38,
                53,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                75,
                90,
                89,
                88,
                87,
                86,
                85,
                84,
                83,
                82,
                81,
                80,
                79,
                78,
                77,
                76,
                91,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                113,
                128,
                127,
                126,
                125,
                124,
                123,
                122,
                121,
                120,
                119,
                118,
                117,
                116,
                115,
                114,
                129,
            ]
        ]
        vert_chain = [
            [
                129,
                114,
                115,
                116,
                110,
                97,
                96,
                95,
                91,
                76,
                77,
                78,
                72,
                59,
                58,
                57,
                53,
                38,
                39,
                40,
                34,
                21,
                20,
                19,
                15,
                0,
                1,
                2,
                3,
                4,
                16,
                23,
                24,
                25,
                35,
                44,
                43,
                42,
                54,
                61,
                62,
                63,
                73,
                82,
                81,
                80,
                92,
                99,
                100,
                101,
                111,
                120,
                119,
                118,
                130,
            ],
            [
                131,
                122,
                123,
                124,
                112,
                105,
                104,
                103,
                93,
                84,
                85,
                86,
                74,
                67,
                66,
                65,
                55,
                46,
                47,
                48,
                36,
                29,
                28,
                27,
                17,
                8,
                9,
                10,
                11,
                12,
                18,
                31,
                32,
                33,
                37,
                52,
                51,
                50,
                56,
                69,
                70,
                71,
                75,
                90,
                89,
                88,
                94,
                107,
                108,
                109,
                113,
                128,
                127,
                126,
                132,
            ],
        ]
    elif backend.num_qubits == 156:
        horz_chain = [
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                19,
                35,
                34,
                33,
                32,
                31,
                30,
                29,
                28,
                27,
                26,
                25,
                24,
                23,
                22,
                21,
                20,
            ],
            [
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                59,
                75,
                74,
                73,
                72,
                71,
                70,
                69,
                68,
                67,
                66,
                65,
                64,
                63,
                62,
                61,
                60,
            ],
            [
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                99,
                115,
                114,
                113,
                112,
                111,
                110,
                109,
                108,
                107,
                106,
                105,
                104,
                103,
                102,
                101,
                100,
            ],
            [
                120,
                121,
                122,
                123,
                124,
                125,
                126,
                127,
                128,
                129,
                130,
                131,
                132,
                133,
                134,
                135,
                139,
                155,
                154,
                153,
                152,
                151,
                150,
                149,
                148,
                147,
                146,
                145,
                144,
                143,
                142,
                141,
                140,
            ],
        ]
        vert_chain = [
            [
                0,
                1,
                2,
                3,
                16,
                23,
                22,
                21,
                36,
                41,
                42,
                43,
                56,
                63,
                62,
                61,
                76,
                81,
                82,
                83,
                96,
                103,
                102,
                101,
                116,
                121,
                122,
                123,
                136,
                143,
                142,
                141,
                140,
            ],
            [
                5,
                6,
                7,
                17,
                27,
                26,
                25,
                37,
                45,
                46,
                47,
                57,
                67,
                66,
                65,
                77,
                85,
                86,
                87,
                97,
                107,
                106,
                105,
                117,
                125,
                126,
                127,
                137,
                147,
                146,
                145,
            ],
            [
                9,
                10,
                11,
                18,
                31,
                30,
                29,
                38,
                49,
                50,
                51,
                58,
                71,
                70,
                69,
                78,
                89,
                90,
                91,
                98,
                111,
                110,
                109,
                118,
                129,
                130,
                131,
                138,
                151,
                150,
                149,
            ],
            [
                13,
                14,
                15,
                19,
                35,
                34,
                33,
                39,
                53,
                54,
                55,
                59,
                75,
                74,
                73,
                79,
                93,
                94,
                95,
                99,
                115,
                114,
                113,
                119,
                133,
                134,
                135,
                139,
                155,
                154,
                153,
            ],
        ]
    else:
        raise ValueError("No grids defined for qubit number %d" % backend.nqubits)

    grid_chains = [horz_chain, vert_chain]

    return grid_chains


def df_to_error_dict(lf_df: DataFrame, gate_list: list):
    """Take in the dataframe from the layer fidelity experiment
    and convert it to a dictionary of gate errors

    Args:
    - lf_df: Dataframe from layer fidelity experiment
    - gate_list: List of gates in the layer fidelity experiment

    Return:
    - error dictionary
    """

    pfdf = lf_df[lf_df.name == "ProcessFidelity"]
    err_rb_dict = {}

    for gate in gate_list:
        gate_pf = pfdf[pfdf.qubits == tuple(gate)].iloc[0].value.nominal_value

        err_rb_dict["%d_%d" % (gate[0], gate[1])] = 4 / 5 * (1 - gate_pf)

    return err_rb_dict


def make_error_dict(backend: IBMBackend, two_q_gate: str, keep_perm: bool = False):
    """Convert a backend into a dictionary of errors for each edge

    Args:
    - backend: ibm backend
    - two_q_gate: the two qubit gate type
    - keep_perm: include the errors if both directions of the gate
    appear in the coupling map

    Return:
    - error dictionary
    """

    err_dict = {}
    props = backend.properties()
    coup_map = backend.coupling_map

    for i in coup_map:
        if not keep_perm:
            if "%d_%d" % (i[1], i[0]) in err_dict:
                continue

        err_dict["%d_%d" % (i[0], i[1])] = props.gate_error(two_q_gate, i)

    return err_dict


def update_error_dict(
    error_dict: dict,
    new_error_dicts: dict,
    update_perm: bool = True,
):
    """Update errors in error_dict with the errors from the
    new_error_dict

    Args:
    - error_dict: original error dictionary (assume complete)
    - new_error_dicts: list of new error dictionaries
    - update_perm: treat Q0_Q1 the same as Q1_Q0

    Return:
    - updated error dictionary
    """

    for gate in error_dict:
        updated_errs = []
        for i in new_error_dicts:
            if gate in i:
                updated_errs.append(i[gate])

            if update_perm:
                rgate = "%s_%s" % (gate.split("_")[1], gate.split("_")[0])
                if rgate in i:
                    updated_errs.append(i[rgate])

        if len(updated_errs) > 0:
            error_dict[gate] = float(np.mean(updated_errs))

    return error_dict


def layer_fid_chain(error_dict, chain):
    """
    Take in a error dictionary and chain and calculate layer fid

    Args:
        error_dict: a dictionary of gate pairs and errors {'1_2': 0.001, '3_5': 0.02} and/or
        single qubit and errors {'1': 0.001}
        chain: 1D chain of qubits
    Returns:
        layer fidelity (depolarizing approx)
    """

    fid_layer = 1

    for qind, q in enumerate(chain):
        # treat the beginning and end differently
        if qind == 0 or qind == (len(chain) - 1):
            if "%d" % q in error_dict:
                fp = 1 - 3 / 2 * error_dict["%d" % q]
                fid_layer *= fp

        if qind == (len(chain) - 1):
            continue

        if "%d_%d" % (q, chain[qind + 1]) in error_dict:
            key_ind = "%d_%d" % (q, chain[qind + 1])
        else:
            key_ind = "%d_%d" % (chain[qind + 1], q)

        # simple gamma calc for depolarizing errors
        fp = 1 - ((2 ** (2) + 1) / 2 ** (2)) * error_dict[key_ind]

        fid_layer *= fp

    return fid_layer


def best_chain(
    nq, coupling_map, error_dict, path_len=10, best_fid_guess=0.95, fid_cutoff=0.9
):
    """
    For a particular coupling map and error directionary compute the best
    chains **heuristically** using DFS
    Warning: this can be very time consuming if the wrong settings are used
    Length 100 chain on eagle, with fid_cutoff=0.95 should take about a minute

    Args:
        nq: number of qubits
        coupling_map: coupling map for the backend (list)
        error_dict: a dictionary of gate pairs and errors {'1_2': 0.001, '3_5': 0.02} and/or
        single qubit and errors {'1': 0.001}
        path_len: Length of the chains to find
        best_fid_guess: initial guess at the best fidelity.
        fid_cutoff: Will reject any paths that don't seem to be better than fid_cutoff*best_fid
        If this is set to 0, ALL paths will be found, but this is very time consuming
        If this is set to, e.g., 0.99, then it will heuristically restrict the paths
        and the time will shorter but not guaranteed to find the best path
    Returns:
        chains: chains found (sorted best to worst)
        fids: fidelity of chains found (sorted best to worst)
    """

    best_fid = [best_fid_guess]

    # calculate all the gammas from the reported gate errors
    len_sets_fid = []
    len_sets_all = []

    # create this graph dictionary of each qubit and it's neighbors
    graph_dict = gu.create_graph_dict(coupling_map, nq)

    for j in range(nq):
        start_q = j
        len_sets = gu.iter_neighbors(
            graph_dict,
            start_q,
            error_dict,
            best_fid,
            fid_cutoff,
            [start_q],
            1.0,
            path_len,
        )

        for i in range(len(len_sets)):
            len_sets[i].reverse
            if len_sets[i] in len_sets_all:
                continue
            len_sets_all.append(len_sets[i])
            len_sets_fid.append(layer_fid_chain(error_dict, len_sets[i]))

    ind_sort = np.argsort(len_sets_fid)
    return np.array(len_sets_all)[ind_sort], np.array(len_sets_fid)[ind_sort]


def grid_chain_to_layers(backend, coupling_map):
    """
    Turn the grid chains into a set of n disjoint layers

    Args:
        backend
        coupling_map
    Returns:
        grid_chain_flt
        layers
    """

    # first get the grid chains, these are hard coded in the layer fidelity utilities module
    grid_chains = get_grids(backend)

    # there are two sets of chains that can be run in four disjoint experiments
    print("Decomposing grid chain into disjoint layers")
    layers = [[] for i in range(4)]
    grid_chain_flt = [[], []]
    for i in range(2):
        all_pairs = gu.path_to_edges(grid_chains[i], coupling_map)
        for j, pair_lst in enumerate(all_pairs):
            grid_chain_flt[i] += grid_chains[i][j]
            sub_pairs = [
                tuple(pair) for pair in pair_lst
            ]  # make this is a list of tuples
            layers[2 * i] += sub_pairs[0::2]
            layers[2 * i + 1] += sub_pairs[1::2]

    # Check that each list is in the coupling map and is disjoint
    for layer in layers:
        for qpair in layer:
            if tuple(qpair) not in coupling_map:
                raise ValueError(f"Gate on {qpair} does not exist")

            for k in layer:
                if k == qpair:
                    continue

                if k[0] in qpair or k[1] in qpair:
                    print(f"Warning: sets are not disjoint for gate {k} and {qpair}")

    return grid_chain_flt, layers
