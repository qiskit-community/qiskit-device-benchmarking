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
from typing import Dict, List, Tuple
import datetime

from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity
from qiskit.transpiler import CouplingMap
import qiskit_device_benchmarking.utilities.graph_utils as gu

def run_lf_chain(
    chain: List[int],
    backend: IBMBackend,
    nseeds: int=6,
    seed: int=42,
    cliff_lengths: List[int]=[1, 10, 20, 30, 40, 60, 80, 100, 150, 200, 400],
    nshots: int=200
) -> ExperimentData:
    
    """ General function to run LF on a list of chains using default or
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
        twoq_gate = 'cx'
    
    # Get one qubit basis gates
    oneq_gates = []
    for i in backend.configuration().basis_gates:
        if i.casefold()!=twoq_gate.casefold():
                oneq_gates.append(i)
    
    # Get the coupling_map
    coupling_map = CouplingMap(backend.configuration().coupling_map)

    # Get graph from coupling map
    G = coupling_map.graph

    # Decompose chain into trivial two disjoint layers (list of list of gates)
    print('Decomposing qubit chain into two disjoint layers')
    all_pairs = gu.path_to_edges([chain], coupling_map)[0]
    layers = [all_pairs[0::2], all_pairs[1::2]] # will run a layer for each list

    # Check that each list is in the coupling map and is disjoint
    for layer in layers:
        for qpair in layer:
            if tuple(qpair) not in coupling_map:
                raise ValueError(f"Gate on {qpair} does not exist")

            for k in layer:
                if k == qpair:
                    continue

                if k[0] in qpair or k[1] in qpair:
                    print(f'Warning: sets are not disjoint for gate {k} and {qpair}')
    print('Disjoint layer 1')
    print(layers[0])
    print('Disjoint layer 2')
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
    print(f"Runing layer fidelity experiment: ID={exp_data.experiment_id} with jobs {exp_data.job_ids}]")
    return exp_data

def get_rb_data(exp_data: ExperimentData)-> pd.DataFrame:
    """ Extract raw and fitted RB data from a layer fidelity experiment.
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
                qubits = (q1,q2)
            # Retrieve by ID
            scatter_table = exp_data.artifacts(curve_data_id).data
            data_df = scatter_table.dataframe
            data_df['qubits'] = [qubits] * len(data_df)
            rb_data_df = pd.concat([rb_data_df, data_df], ignore_index=True)
    return rb_data_df

def reconstruct_lf_per_length(exp_data: ExperimentData, qchain: List[int], backend: IBMBackend) -> pd.DataFrame:  
    """ Extracts the lf and eplg values from a layer fidelity experiment for all 
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
    # Get the coupling_map
    G = coupling_map.graph

    # Recover layers from experiment
    all_pairs = gu.path_to_edges([qchain], coupling_map)[0]
    layers = [all_pairs[0::2], all_pairs[1::2]]  # will run a layer for each list
    full_layer = [None] * (len(layers[0]) + len(layers[1]))
    full_layer[::2] = layers[0]
    full_layer[1::2] = layers[1]
    full_layer = [(qchain[0],)] + full_layer + [(qchain[-1],)]

    # Compute LF by chain length assuming the first layer is full with 2q-gates
    df = exp_data.analysis_results(dataframe=True)
    pfdf = df[df.name == "ProcessFidelity"]
    pfdf = pfdf.fillna({"value": 0}) # Fill Process Fidelity nan values with zeros
    results_per_chain = []
    # Check if the dataframe is empty
    if len(pfdf) > 0:
        # pfs[i] corresponds to edges[i]
        pfs = [pfdf.loc[pfdf[pfdf.qubits == qubits].index[0], 'value'] for qubits in full_layer]
        pfs = list(map(lambda x: x.n if x != 0 else 0, pfs))
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
                np.sqrt(pfs[s]) * np.prod(pfs[s + 1 : s + w - 1]) * np.sqrt(pfs[s + w - 1])
                for s in range(len(pfs) - w + 1)
            )
            fid_w = []
            chains = []
            for s in range(len(pfs) - w + 1):
                fid_w.append(np.sqrt(pfs[s]) * np.prod(pfs[s + 1 : s + w - 1]) * np.sqrt(pfs[s + w - 1]))
                chains.append(qchain[s:s+w-1])

            chain = chains[np.argmax(fid_w)] # Get chain of length 'length' with highest fidelity
            chain_fids.append(np.max(fid_w)) # save highest fidelity
            lens = np.arange(4, length + 1, 1)

            # Get EPLG by chain length
            num_2q_gates = [l - 1 for l in lens]
            chain_eplgs = [
                1 - (fid ** (1 / num_2q)) for num_2q, fid in zip(num_2q_gates, chain_fids)
            ]
            results_per_chain.append(
                {
                    'qchain_length': len(chain),
                    'qchain': list(chain),
                    'lf': np.asarray(chain_fids),
                    'eplg': np.asarray(chain_eplgs),
                    'length': np.asarray(lens),
                    'job_ids': exp_data.job_ids
                }
            )
    # Save data
    results_per_chain = pd.DataFrame.from_dict(results_per_chain)
    return results_per_chain
    

def make_lf_eplg_plots(
    exp_data: ExperimentData,
    chain: List[int],
    machine: str
):
    """ Make layer fidelity and eplg plots for a given chain. Values
    are extracted from the provided experiment object.
    
    Args:
    - exp_data: ExperimentData object
    - chain: list of qubits in chain
    - machine: name of the backend
    """
    # Get the coupling_map
    coupling_map = CouplingMap(backend.configuration().coupling_map)
    # Get the coupling_map
    G = coupling_map.graph

    # cover layers from experiment
    all_pairs = gu.path_to_edges([chain], coupling_map)[0]
    layers = [all_pairs[0::2], all_pairs[1::2]]  # will run a layer for each list
    full_layer = [None] * (len(layers[0]) + len(layers[1]))
    full_layer[::2] = layers[0]
    full_layer[1::2] = layers[1]
    full_layer = [(chain[0],)] + full_layer + [(chain[-1],)]
    if len(full_layer) != len(chain) + 1:
        print(f'Length of full layer (= {len(full_layer)}) is not equal to length of the chain + 1 (= {len(chain) + 1})')
        print('Make su the specified qubit chain is consistent with the device map')
        print(f'Full layer is {full_layer}')
        print(f'Chain is {chain}')
        exit()

    # Compute LF by chain length assuming the first layer is full with 2q-gates
    df = exp_data.analysis_results(dataframe=True)
    pfdf = df[df.name == "ProcessFidelity"]
    pfdf = pfdf.fillna({"value": 0}) # Fill Process Fidelity nan values with zeros
    sults_per_chain = []
    # Check if the dataframe is empty
    if len(pfdf) > 0:
        pfs = [pfdf.loc[pfdf[pfdf.qubits == qubits].index[0], 'value'] for qubits in full_layer]
        pfs = list(map(lambda x: x.n if x != 0 else 0, pfs))
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
                np.sqrt(pfs[s]) * np.prod(pfs[s + 1 : s + w - 1]) * np.sqrt(pfs[s + w - 1])
                for s in range(len(pfs) - w + 1)
            )
            chain_fids.append(fid_w)

        # Get EPLG by chain length
        num_2q_gates = [length - 1 for length in chain_lens]
        chain_eplgs = [
            1 - (fid ** (1 / num_2q)) for num_2q, fid in zip(num_2q_gates, chain_fids)
        ]
        sults_per_chain.append(
            {
                'qchain': list(chain),
                'lf': np.asarray(chain_fids),
                'eplg': np.asarray(chain_eplgs),
                'length': np.asarray(chain_lens),
                'job_ids': exp_data.job_ids
            }
        )
    
    time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    # Plot LF by chain length
    fig1, ax1 = plt.subplots(figsize=(8.5, 4))
    title_lf = f"{machine} {time} LF for fixed {len(chain)}q chains"
    label_lf = f'Lf: {np.round(chain_fids[-1],3)}'
    ax1.plot(chain_lens, chain_fids, marker="o", linestyle="-", label=label_lf) # lfs
    ax1.set_title(title_lf, fontsize=11)
    ax1.set_xlim(0, chain_lens[-1] * 1.05)
    ax1.set_ylim(0.95 * min(chain_fids), 1)
    ax1.set_ylabel("Layer Fidelity (LF)")
    ax1.set_xlabel("Chain Length")
    ax1.grid()
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax1.figu.set_dpi(150)
    fig1.tight_layout()
    fig1.savefig(f'{machine}_lf.png')
    
    # Plot EPLG by chain length
    fig2, ax2 = plt.subplots(figsize=(8.5, 4))
    label_eplg = f'Eplg: {np.round(chain_eplgs[-1],3)}'
    title_eplg = f"{machine} {time} EPLG for fixed {len(chain)}q chains"
    ax2.plot(chain_lens, chain_eplgs, marker="o", linestyle="-", label=label_eplg) # eplgs
    ax2.set_title(title_eplg, fontsize=11)
    ax2.set_xlim(0, chain_lens[-1] * 1.05)
    ax2.set_ylabel("Error per Layed Gate (EPLG)")
    ax2.set_xlabel("Chain Length")
    ax2.grid()
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax2.figu.set_dpi(150)
    fig2.tight_layout()
    fig2.savefig(f'{machine}_eplg.png')
