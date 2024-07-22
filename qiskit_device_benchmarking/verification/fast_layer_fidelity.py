#!/usr/bin/env python
# This code is part of Qiskit.
#
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
Fast Layer Fidelity on the reported 100Q qiskit chain
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
import rustworkx as rx
import pandas as pd
from typing import Dict, List, Tuple
import datetime
import yaml

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity
from qiskit.visualization import plot_gate_map
from qiskit.transpiler import CouplingMap

def run_fast_lf(backends: List[str],
    nseeds: int,
    seed: int,
    cliff_lengths: List[int],
    nshots: int,
    act_name: str,
    hgp: str):
    
    # load the service
    print('Loading service')
    service = QiskitRuntimeService(name=act_name)
    
    for backend_name in backends:
        print(f'Current bakend is {backend_name}')
        # Get the real backend
        backend = service.backend(backend_name, instance=hgp)
        
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

        # Get 100Q chain from qiskit
        qchain = backend.properties().general_qlists[0]['qubits']
        print(f'100Q chain for {backend_name} is: ', qchain)

        # Run LF
        print(f'Running LF on {backend_name}')
        exp_data = run_lf_chain(
            chain=qchain,
            backend=backend,
            nseeds=nseeds,
            seed=seed,
            cliff_lengths=cliff_lengths,
            nshots=nshots,
            twoq_gate=twoq_gate,
            oneq_gates=oneq_gates,
            coupling_map=coupling_map)

        # Fit 2Q experiment data
        print(f'Retrieving experiment results from {backend_name}')
        exp_data.block_for_results()

        # Get experiment data as df
        df = exp_data.analysis_results(dataframe=True)

        # Retrieve raw and fitted RB data
        rb_data_df = pd.DataFrame()
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

        # Save RB data locally
        print(f'Saving 2Q data from {backend_name}')
        rb_data_df.to_csv(f'{backend_name}_full_rb_data.csv', float_format='%.15f')

        # Plot LF and EPLG data
        print(f'Making plots for {backend_name}')
        make_lf_eplg_plots(
            chain=qchain,
            exp_data=exp_data,
            machine=backend_name,
            coupling_map=coupling_map,
            df=df
        )


def to_edges(
    path: List[int], coupling_map: CouplingMap, G: rx.PyDiGraph
) -> List[Tuple[int]]:
    """
    Return a list of edges from a path of qubits.
    Args:
    - path: List of nodes (qubits)
    - coupling_map: map containing connections
    - G: graph
    Returns:
    - List of edges
    """

    edges = []
    prev_node = None
    # Get all prossible edges from the nodes provided
    for node in path:
        if prev_node is not None:
            # Check the edge is in G and in the coupling map
            if G.has_edge(prev_node, node) and (prev_node, node) in coupling_map:
                edges.append((prev_node, node))
            elif G.has_edge(node, prev_node) and (node, prev_node) in coupling_map:
                edges.append((node, prev_node))
        prev_node = node
    return edges


def run_lf_chain(
    chain: List[int],
    backend: IBMBackend,
    nseeds: int,
    seed: int,
    cliff_lengths: List[int],
    nshots: int,
    twoq_gate: str,
    oneq_gates: List[str],
    coupling_map: CouplingMap,
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
    - twoq_gate: type of two qubit gate 
    - oneq_gates: list one qubits gates
    Returns:
    - List of ExperimentData objects
    """

    # Get graph from coupling map
    G = coupling_map.graph

    # Decompose chain into trivial two disjoint layers (list of list of gates)
    all_pairs = to_edges(chain, coupling_map, G)
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
    print(f"Run experiment: ID={exp_data.experiment_id} with jobs {exp_data.job_ids}]")
    return exp_data


def make_lf_eplg_plots(
    chain: List[int],
    exp_data: ExperimentData,
    machine: str,
    coupling_map: CouplingMap,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract experiment results for a list of chains and create a DataFrame
    with LF and EPLG values.
    Args:
    - best_qubit_chains: List of chains
    - lf_experiments_best: List of LF experiments
    - backend
    Returns:
    - DataFrame containing experiment results per chain
    """

    # Get the coupling_map
    G = coupling_map.graph

    # Recover layers from experiment
    all_pairs = to_edges(chain, coupling_map, G)
    layers = [all_pairs[0::2], all_pairs[1::2]]  # will run a layer for each list
    full_layer = [None] * (len(layers[0]) + len(layers[1]))
    full_layer[::2] = layers[0]
    full_layer[1::2] = layers[1]
    full_layer = [(chain[0],)] + full_layer + [(chain[-1],)]
    if len(full_layer) != len(chain) + 1:
        print(f'Length of full layer (= {len(full_layer)}) is not equal to length of the chain + 1 (= {len(chain) + 1})')
        print('Make sure the specified qubit chain is consistent with the device map')
        print(f'Full layer is {full_layer}')
        print(f'Chain is {chain}')
        exit()

    # Compute LF by chain length assuming the first layer is full with 2q-gates
    pfdf = df[df.name == "ProcessFidelity"]
    pfdf = pfdf.fillna({"value": 0}) # Fill Process Fidelity nan values with zeros
    results_per_chain = []
    # Check if the dataframe is empty
    if len(pfdf) > 0:
        pfs = [pfdf.loc[pfdf[pfdf.qubits == qubits].index[0], 'value'] for qubits in full_layer]
        pfs = list(map(lambda x: x.n if x != 0 else 0, pfs))
        pfs[0] = pfs[0] ** 2
        pfs[-1] = pfs[-1] ** 2

        # Approximate 1Q RB fidelities at both ends by the square root of 2Q RB fidelity at
        # both ends. For example, if we have [(0, 1), (1, 2), (2, 3), (3, 4)] 2Q RB fidelities
        # and if we want to compute a layer fidelity for [1, 2, 3], we approximate the 1Q
        # filedities for (1,) and (3,) by the square root of 2Q fidelities of (0, 1) and (3, 4).
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
        results_per_chain.append(
            {
                'qchain': list(chain),
                'lf': np.asarray(chain_fids),
                'eplg': np.asarray(chain_eplgs),
                'length': np.asarray(chain_lens),
                'job_ids': exp_data.job_ids
            }
        )
        
    # Save data
    results_per_chain = pd.DataFrame.from_dict(results_per_chain)
    results_per_chain.to_csv(f'{machine}_lf_eplg_data.csv', float_format='%.15f')
    
    # Make plots
    time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    fig1, ax1 = plt.subplots(figsize=(8.5, 4))
    fig2, ax2 = plt.subplots(figsize=(8.5, 4))
    
    # Plot LF by chain length
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
    ax1.figure.set_dpi(150)
    fig1.tight_layout()
    fig1.savefig(f'{machine}_lf.png')
    
    # Plot EPLG by chain length
    label_eplg = f'Eplg: {np.round(chain_eplgs[-1],3)}'
    title_eplg = f"{machine} {time} EPLG for fixed {len(chain)}q chains"
    ax2.plot(chain_lens, chain_eplgs, marker="o", linestyle="-", label=label_eplg) # eplgs
    ax2.set_title(title_eplg, fontsize=11)
    ax2.set_xlim(0, chain_lens[-1] * 1.05)
    ax2.set_ylabel("Error per Layered Gate (EPLG)")
    ax2.set_xlabel("Chain Length")
    ax2.grid()
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax2.figure.set_dpi(150)
    fig2.tight_layout()
    fig2.savefig(f'{machine}_eplg.png')

def import_yaml(fstr):
    with open(fstr, 'r') as stream:
        data_imp = yaml.safe_load(stream)
        
    return data_imp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run fast layer fidelity '
                                     + 'on reported qikist chain. Specify a config '
                                     +' yaml and override settings on the command line')
    parser.add_argument('-c', '--config', help='config file name', 
                        default='config.yaml')
    parser.add_argument('-b', '--backend', help='Specify backend and override '
                        + 'backend_group')
    parser.add_argument('-bg', '--backend_group', 
                        help='specify backend group in config file', 
                        default='backends')
    parser.add_argument('--hgp', help='specify hgp')
    parser.add_argument('--name', help='Account name', default='')
    parser.add_argument('--nseeds', help='number of seeds', default=6)
    parser.add_argument('--seed', help='seed to use', default=42)
    parser.add_argument('--nshots', help='number of shots', default=200)
    parser.add_argument(
        '--cliff_lengths',
        help='list of clifford lenghts [...]',
        default=[1, 10, 20, 30, 40, 60, 80, 100, 150, 200, 400],
    )
    args = parser.parse_args()
    
    #import from config
    config_dict = import_yaml(args.config)
    print('Config File Found')
    print(config_dict)
    
    #override from the command line
    if args.backend is not None:
        backends = [args.backend]
    else:
        backends = config_dict[args.backend_group]  
    if args.hgp is not None:
        hgp = args.hgp
    else:
        hgp = config_dict['hgp']   
    # set default values unless otherwise instructed on config_dict 
    if 'nseeds' in config_dict.keys():
        nseeds = config_dict['nseeds']
    else:
        nseeds = args.nseeds   
    if 'seed' in config_dict.keys():
        seed = config_dict['seed']
    else:
        seed = args.seed   
    if 'cliff_lengths' in config_dict.keys():
        cliff_lengths = config_dict['cliff_lengths']
    else:
        cliff_lengths = args.cliff_lengths   
    if 'nshots' in config_dict.keys():
        nshots = config_dict['nshots']
    else:
        nshots = args.nshots
    if 'act_name' in config_dict.keys():
        act_name = config_dict['act_name']
    else:
        act_name = args.name
    
    # Run fast layer fidelity on the list of backends
    print('Running fast layer fidelity on backend(s)')
    run_fast_lf(backends, nseeds, seed, cliff_lengths, nshots, act_name, hgp)
