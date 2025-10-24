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
Make a new error map by running layer fidelity on a grid of 
qubits
"""

import argparse
import warnings
from qiskit_ibm_runtime import QiskitRuntimeService

import qiskit_device_benchmarking.utilities.file_utils as fu
import qiskit_device_benchmarking.utilities.graph_utils as gu
import qiskit_device_benchmarking.utilities.layer_fidelity_utils as lfu
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity

def run_grid(
    backend_name,
    trials=3,
    nshots=200,
    act_name="",
    seed=42,
    write_file=""
):
    """Run a layer fidelity on a grid


    Args:
        backend_name: backend to run on
        trials: number of randomizations
        nshots: number of shots
        act_name: account name to be passed to the runtime service
        seed: randomization seed
        write_file: file to write the new error map to

    Returns:
        prints the dictionary to the cmd line
    """

    warnings.filterwarnings(
        "error", message=".*run.*", category=DeprecationWarning, append=False
    )

    # load the service
    service = QiskitRuntimeService(name=act_name)
    backend = service.backend(backend_name, use_fractional_gates=False)
    coupling_map = backend.coupling_map

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
    
    # first get the grid chains, these are hard coded in the layer fidelity utilities module
    grid_chains = lfu.get_grids(backend)

    # there are two sets of chains that can be run in four disjoint experiments
    print("Decomposing grid chain into disjoint layers")
    layers = [[] for i in range(4)]
    grid_chain_flt = [[], []]
    for i in range(2):
        all_pairs = gu.path_to_edges(grid_chains[i], coupling_map)
        for j, pair_lst in enumerate(all_pairs):
            grid_chain_flt[i] += grid_chains[i][j]
            sub_pairs = [tuple(pair) for pair in pair_lst]  # make this is a list of tuples
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


    # generate two experiments
    lfexps = []
    for i in range(2):
        lfexps.append(
            LayerFidelity(
                physical_qubits=grid_chain_flt[i],
                two_qubit_layers=layers[2 * i : (2 * i + 2)],
                lengths=[1, 10, 20, 30, 40, 60, 80, 100, 150, 200, 400],
                backend=backend,
                num_samples=trials,
                seed=seed,
                two_qubit_gate=twoq_gate,
                one_qubit_basis_gates=oneq_gates,
            )
        )

    # set maximum number of circuits per job to avoid errors due to too large payload
    lfexps[i].experiment_options.max_circuits = 300

    # Run the LF experiment (generate circuits and submit the job)
    exp_data_lst = []
    for i in range(2):
        exp_data_lst.append(lfexps[i].run(shots=nshots))
        print(
            f"Run experiment: ID={exp_data_lst[i].experiment_id} with jobs {exp_data_lst[i].job_ids}]"
        )

    # make an updated error map
    updated_err_dicts = []
    err_dict = lfu.make_error_dict(backend, twoq_gate)

    for i in range(2):
        # get the results from the experiment
        df = exp_data_lst[i].analysis_results(dataframe=True)
        for j in range(2):
            updated_err_dicts.append(lfu.df_to_error_dict(df, layers[2 * i + j]))


    err_dict = lfu.update_error_dict(err_dict, updated_err_dicts)

    err_dict_ex = {'err_dict': err_dict, 'config': {}}

    err_dict_ex['config']['backend'] = backend_name
    err_dict_ex['config']['jobs'] = [exp_data_lst[i].job_ids for i in range(2)]
    err_dict_ex['config']['samples'] = trials
    err_dict_ex['config']['seed'] = seed
    err_dict_ex['config']['nshots'] = nshots

    fu.export_yaml(write_file, err_dict_ex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run layer fidelity on a  "
        + "prespecified grid of qubits to "
        + " generate a new error map yaml"
    )
    parser.add_argument(
        "-b", "--backend", help="Specify backend", required=True
    )
    parser.add_argument("--name", help="Account name", default="")
    parser.add_argument("--nshots", help="Number of shots to use", default=200)
    parser.add_argument("--trials", help="Number of randomizations", default=3)
    parser.add_argument("--seed", help="Seed to use", default=42)
    parser.add_argument("--file", help="file to write to")
    args = parser.parse_args()

    if args.file is not None:
        out_file = args.file
    else:
        out_file = "%s_grid_err_"%args.backend + fu.timestamp_name() + ".yaml"

    run_grid(
        backend_name=args.backend,
        trials=args.trials,
        nshots=args.nshots,
        act_name=args.name,
        seed=args.seed,
        write_file=out_file
    )
