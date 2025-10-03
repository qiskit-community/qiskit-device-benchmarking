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
Fast benchmark of qubit count using the CHSH inequality
"""

import argparse
import rustworkx as rx
import networkx as nx
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_experiments.framework import ParallelExperiment, BatchExperiment

import qiskit_device_benchmarking.utilities.file_utils as fu
import qiskit_device_benchmarking.utilities.graph_utils as gu
from qiskit_device_benchmarking.bench_code.bell import CHSHExperiment


def run_count(hgp, backends, nshots=100, act_name=""):
    """Run a chsh inequality on a number of devices

    Args:
        hgp: hub/group/project
        backends: list of backends
        nshots: number of shots
        act_name: account name to be passed to the runtime service

    Returns:
        flat list of all the edges
    """

    # load the service
    service = QiskitRuntimeService(name=act_name)
    job_list = []
    result_dict = {}
    result_dict["config"] = {"hgp": hgp, "nshots": nshots, "act_name": act_name}

    print("Running Fast Count with options %s" % result_dict["config"])

    # run all the circuits
    for backend in backends:
        print("Loading backend %s" % backend)
        result_dict[backend] = {}
        backend_real = service.backend(backend, instance=hgp)
        chsh_exp_list_b = []

        # compute the sets for this
        # NOTE: I want to replace this with fixed sets from
        # a config file!!!
        nq = backend_real.configuration().n_qubits
        coupling_map = backend_real.configuration().coupling_map
        # build a set of gates
        G = gu.build_sys_graph(nq, coupling_map)
        # get all length 2 paths in the device
        paths = rx.all_pairs_all_simple_paths(G, 2, 2)
        # flatten those paths into a list from the rustwork x iterator
        paths = gu.paths_flatten(paths)
        # remove permutations
        paths = gu.remove_permutations(paths)
        # convert to the coupling map of the device
        paths = gu.path_to_edges(paths, coupling_map)
        # make into separate sets
        sep_sets = gu.get_separated_sets(G, paths, min_sep=2)

        result_dict[backend]["sets"] = sep_sets

        # Construct mirror QV circuits on each parallel set
        for qsets in sep_sets:
            chsh_exp_list = []

            for qset in qsets:
                # generate the circuits
                chsh_exp = CHSHExperiment(physical_qubits=qset, backend=backend_real)

                chsh_exp.set_transpile_options(optimization_level=1)
                chsh_exp_list.append(chsh_exp)

            new_exp_chsh = ParallelExperiment(
                chsh_exp_list, backend=backend_real, flatten_results=False
            )

            chsh_exp_list_b.append(new_exp_chsh)

        new_exp_chsh = BatchExperiment(
            chsh_exp_list_b, backend=backend_real, flatten_results=False
        )

        new_exp_chsh.set_run_options(shots=nshots)
        job_list.append(new_exp_chsh.run())
        result_dict[backend]["job_ids"] = job_list[-1].job_ids

    # get the jobs back
    for i, backend in enumerate(backends):
        print("Loading results for backend: %s" % backend)

        expdata = job_list[i]
        try:
            expdata.block_for_results()
        except Exception:
            # remove backend from results
            print("Error loading backend %s results" % backend)
            result_dict.pop(backend)
            continue

        result_dict[backend]["chsh_values"] = {}

        for qsets_i, qsets in enumerate(result_dict[backend]["sets"]):
            for qset_i, qset in enumerate(qsets):
                anal_res = (
                    expdata.child_data()[qsets_i]
                    .child_data()[qset_i]
                    .analysis_results()[0]
                )
                qedge = "%d_%d" % (
                    anal_res.device_components[0].index,
                    anal_res.device_components[1].index,
                )
                result_dict[backend]["chsh_values"][qedge] = anal_res.value

        # calculate number of connected qubits
        G = nx.Graph()

        # add all possible edges
        for i in result_dict[backend]["chsh_values"]:
            if result_dict[backend]["chsh_values"][i] >= 2:
                G.add_edge(int(i.split("_")[0]), int(i.split("_")[1]))

        # catch error if the graph is empty
        try:
            largest_cc = max(nx.connected_components(G), key=len)

            # look at the average degree of the largest region
            avg_degree = 0
            for i in largest_cc:
                avg_degree += nx.degree(G, i)

            avg_degree = avg_degree / len(largest_cc)

        except Exception:
            largest_cc = {}
            avg_degree = 1

        result_dict[backend]["largest_region"] = len(largest_cc)
        result_dict[backend]["average_degree"] = avg_degree

    fu.export_yaml("CHSH_" + fu.timestamp_name() + ".yaml", result_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fast benchmark of "
        + "qubit count using chsh. Specify a config "
        + " yaml and override settings on the command line"
    )
    parser.add_argument(
        "-c", "--config", help="config file name", default="config.yaml"
    )
    parser.add_argument(
        "-b", "--backend", help="Specify backend and override " + "backend_group"
    )
    parser.add_argument(
        "-bg",
        "--backend_group",
        help="specify backend group in config file",
        default="backends",
    )
    parser.add_argument("--hgp", help="specify hgp")
    parser.add_argument("--shots", help="specify number of shots")
    parser.add_argument("--name", help="Account name", default="")
    args = parser.parse_args()

    # import from config
    config_dict = fu.import_yaml(args.config)
    print("Config File Found")
    print(config_dict)

    # override from the command line
    if args.backend is not None:
        backends = [args.backend]
    else:
        backends = config_dict[args.backend_group]

    if args.hgp is not None:
        hgp = args.hgp
    else:
        hgp = config_dict["hgp"]

    if args.shots is not None:
        nshots = int(args.shots)
    else:
        nshots = config_dict["shots"]

    # print(hgp, backends, he, opt_level, dd, depths, trials, nshots)

    run_count(hgp, backends, nshots=nshots, act_name=args.name)
