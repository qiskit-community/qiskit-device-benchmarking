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
Analyze the benchmarking results
"""

import argparse
import numpy as np
import qiskit_device_benchmarking.utilities.file_utils as fu
import matplotlib.pyplot as plt


def generate_plot(out_data, config_data, args):
    """Generate a plot from the fast_bench data

    Generates a plot of the name result_plot_<XXX>.pdf where XXX is the
    current date and time

    Args:
        out_data: data from the run
        config_data: configuration data from the run
        args: arguments passed to the parser

    Returns:
        None
    """

    markers = ["o", "x", ".", "s", "^", "v", "*"]

    for i, backend in enumerate(out_data):
        plt.semilogy(
            out_data[backend][0],
            out_data[backend][1],
            label=backend,
            marker=markers[np.mod(i, len(markers))],
        )

    plt.legend()
    plt.xlabel("Depth")
    plt.ylabel("Success Probability (%s over sets)" % args.value)
    plt.ylim(top=1.0)
    plt.title(
        "Running Mirror - HE: %s, DD: %s, Trials: %d"
        % (config_data["he"], config_data["dd"], config_data["trials"])
    )
    plt.grid(True)
    plt.savefig("result_plot_%s.pdf" % fu.timestamp_name())
    plt.close()

    return


if __name__ == "__main__":
    """Analyze a benchmarking run from `fast_bench.py`

    Args:
        Call -h for arguments

    """

    parser = argparse.ArgumentParser(
        description="Analyze the results of a " + "benchmarking run."
    )
    parser.add_argument("-f", "--files", help="Comma separated list of files")
    parser.add_argument(
        "-b",
        "--backends",
        help="Comma separated list of " + "backends to plot. If empty plot all.",
    )
    parser.add_argument(
        "-v",
        "--value",
        help="Statistical value to compute",
        choices=["mean", "median", "max", "min"],
        default="mean",
    )
    parser.add_argument("--plot", help="Generate a plot", action="store_true")
    args = parser.parse_args()

    # import from results files and concatenate into a larger results
    results_dict = {}
    for file in args.files.split(","):
        results_dict_new = fu.import_yaml(file)

        for backend in results_dict_new:
            if backend not in results_dict:
                results_dict[backend] = results_dict_new[backend]
            elif backend != "config":
                # backend in the results dict but maybe not that depth
                for depth in results_dict_new[backend]:
                    if depth in results_dict[backend]:
                        err_str = (
                            "Depth %s already exists for backend %s, duplicate results"
                            % (depth, backend)
                        )
                        raise ValueError(err_str)
                    else:
                        # check the metadata is the same
                        # TO DO

                        results_dict[backend][depth] = results_dict_new[backend][depth]

    if args.backends is not None:
        backends_filt = args.backends.split(",")
    else:
        backends_filt = []

    out_data = {}

    for backend in results_dict:
        if len(backends_filt) > 0:
            if backend not in backends_filt:
                continue

        if backend == "config":
            continue
        print(backend)
        depth_list = []
        depth_list_i = []

        out_data[backend] = []

        for depth in results_dict[backend]:
            if depth == "job_ids":
                continue
            depth_list_i.append(depth)
            if args.value == "mean":
                depth_list.append(np.mean(results_dict[backend][depth]["mean"]))
            elif args.value == "max":
                depth_list.append(np.max(results_dict[backend][depth]["mean"]))
            elif args.value == "min":
                depth_list.append(np.min(results_dict[backend][depth]["mean"]))
            else:
                depth_list.append(np.median(results_dict[backend][depth]["mean"]))

        print("Backend %s" % backend)
        print("Depths: %s" % depth_list_i)

        if args.value == "mean":
            print("Means: %s" % depth_list)
        elif args.value == "max":
            print("Max: %s" % depth_list)
        elif args.value == "min":
            print("Min: %s" % depth_list)
        else:
            print("Median: %s" % depth_list)

        out_data[backend].append(depth_list_i)
        out_data[backend].append(depth_list)

    if args.plot:
        generate_plot(out_data, results_dict["config"], args)

    elif args.plot:
        print("Need to run mean/max also")
