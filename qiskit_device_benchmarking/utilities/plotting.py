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
Utilities for plotting
"""

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

    # Recover layers from experiment
    all_pairs = path_to_edges([chain], coupling_map)[0]
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
    df = exp_data.analysis_results(dataframe=True)
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
    ax1.figure.set_dpi(150)
    fig1.tight_layout()
    fig1.savefig(f'{machine}_lf.png')
    
    # Plot EPLG by chain length
    fig2, ax2 = plt.subplots(figsize=(8.5, 4))
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
