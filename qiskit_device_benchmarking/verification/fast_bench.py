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
Fast benchmark via mirror circuits
"""

import argparse
import numpy as np
import rustworkx as rx
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import Target, CouplingMap
from qiskit_experiments.framework import ParallelExperiment, BatchExperiment


import qiskit_device_benchmarking.utilities.file_utils as fu
import qiskit_device_benchmarking.utilities.graph_utils as gu
from qiskit_device_benchmarking.bench_code.mrb import MirrorQuantumVolume

import warnings

from qiskit.circuit import Gate

xslow = Gate(name="xslow", num_qubits=1, params=[])


def run_bench(
    hgp,
    backends,
    depths=[8],
    trials=10,
    nshots=100,
    he=True,
    dd=True,
    opt_level=3,
    act_name="",
):
    """Run a benchmarking test (mirror QV) on a set of devices

    Args:
        hgp: hub/group/project
        backends: list of backends
        depths: list of mirror depths (square circuits)
        trials: number of randomizations
        nshots: number of shots
        he: hardware efficient True/False (False is original QV circ all to all,
                                           True assumes a line)
        dd: add dynamic decoupling
        opt_level: optimization level of the transpiler
        act_name: account name to be passed to the runtime service

    Returns:
        flat list of lists of qubit chains
    """

    warnings.filterwarnings(
        "error", message=".*run.*", category=DeprecationWarning, append=False
    )

    # load the service
    service = QiskitRuntimeService(name=act_name)
    job_list = []
    result_dict = {}
    result_dict["config"] = {
        "hgp": hgp,
        "depths": depths,
        "trials": trials,
        "nshots": nshots,
        "dd": dd,
        "he": he,
        "pregenerated": False,
        "opt_level": opt_level,
        "act_name": act_name,
    }

    print("Running Fast Bench with options %s" % result_dict["config"])

    # run all the circuits
    for backend in backends:
        print("Loading backend %s" % backend)
        result_dict[backend] = {}
        backend_real = service.backend(backend, instance=hgp)
        mqv_exp_list_d = []
        for depth in depths:
            print("Generating Depth %d Circuits for Backend %s" % (depth, backend))

            result_dict[backend][depth] = {}

            # compute the sets for this
            # NOTE: I want to replace this with fixed sets from
            # a config file!!!
            nq = backend_real.configuration().n_qubits
            coupling_map = backend_real.configuration().coupling_map
            G = gu.build_sys_graph(nq, coupling_map)
            paths = rx.all_pairs_all_simple_paths(G, depth, depth)
            paths = gu.paths_flatten(paths)
            new_sets = gu.get_separated_sets(G, paths, min_sep=2, nsets=1)

            mqv_exp_list = []

            result_dict[backend][depth]["sets"] = new_sets[0]

            # Construct mirror QV circuits on each parallel set
            for qset in new_sets[0]:
                # generate the circuits
                mqv_exp = MirrorQuantumVolume(
                    qubits=qset,
                    backend=backend_real,
                    trials=trials,
                    pauli_randomize=True,
                    he=he,
                )

                mqv_exp.analysis.set_options(
                    plot=False, calc_hop=False, analyzed_quantity="Success Probability"
                )

                # Do this so it won't compile outside the qubit sets
                cust_map = []
                for i in coupling_map:
                    if i[0] in qset and i[1] in qset:
                        cust_map.append(i)

                basis_gates = backend_real.configuration().basis_gates
                if "xslow" in basis_gates:
                    basis_gates.remove("xslow")
                if "rx" in basis_gates:
                    basis_gates.remove("rx")
                if "rzz" in basis_gates:
                    basis_gates.remove("rzz")
                cust_target = Target.from_configuration(
                    basis_gates=basis_gates,
                    num_qubits=nq,
                    coupling_map=CouplingMap(cust_map),
                )

                mqv_exp.set_transpile_options(
                    target=cust_target, optimization_level=opt_level
                )
                mqv_exp_list.append(mqv_exp)

            new_exp_mqv = ParallelExperiment(
                mqv_exp_list, backend=backend_real, flatten_results=False
            )
            if dd:
                # this forces the circuits to have DD on them
                print("Transpiling and DD")
                for i in mqv_exp_list:
                    i.dd_circuits()

            mqv_exp_list_d.append(new_exp_mqv)

        new_exp_mqv = BatchExperiment(
            mqv_exp_list_d, backend=backend_real, flatten_results=False
        )
        new_exp_mqv.set_run_options(shots=nshots)
        job_list.append(new_exp_mqv.run())
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

        for j, depth in enumerate(depths):
            result_dict[backend][depth]["data"] = []
            result_dict[backend][depth]["mean"] = []
            result_dict[backend][depth]["std"] = []

            for k in range(len(result_dict[backend][depth]["sets"])):
                result_dict[backend][depth]["data"].append(
                    [
                        float(probi)
                        for probi in list(
                            expdata.child_data()[j].child_data()[k].artifacts()[0].data
                        )
                    ]
                )
                result_dict[backend][depth]["mean"].append(
                    float(np.mean(result_dict[backend][depth]["data"][-1]))
                )
                result_dict[backend][depth]["std"].append(
                    float(np.std(result_dict[backend][depth]["data"][-1]))
                )

    fu.export_yaml("MQV_" + fu.timestamp_name() + ".yaml", result_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fast benchmark of "
        + "devices using mirror. Specify a config "
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
    parser.add_argument("--he", help="Hardware efficient", action="store_true")
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

    if args.he is True:
        he = True
    else:
        he = config_dict["he"]

    opt_level = config_dict["opt_level"]
    dd = config_dict["dd"]
    depths = config_dict["depths"]
    trials = config_dict["trials"]
    nshots = config_dict["shots"]

    # print(hgp, backends, he, opt_level, dd, depths, trials, nshots)

    run_bench(
        hgp,
        backends,
        depths=depths,
        trials=trials,
        nshots=nshots,
        he=he,
        dd=dd,
        opt_level=opt_level,
        act_name=args.name,
    )
