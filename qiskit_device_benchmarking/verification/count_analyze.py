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
Analyze the fast_count results
"""

import argparse
import numpy as np
import qiskit_device_benchmarking.utilities.file_utils as fu
import matplotlib.pyplot as plt


def generate_plot(out_data, degree_data, args):
    """Generate a bar plot of the qubit numbers of each backend

    Generates a plot of the name count_plot_<XXX>.pdf where XXX is the
    current date and time

    Args:
        out_data: data from the run (count data)
        degree_data: average degree
        args: arguments passed to the parser

    Returns:
        None
    """

    count_data = np.array([out_data[i] for i in out_data])
    degree_data = np.array([degree_data[i] for i in out_data])
    backend_lbls = np.array([i for i in out_data])
    sortinds = np.argsort(count_data)

    plt.bar(backend_lbls[sortinds], count_data[sortinds])
    plt.xticks(rotation=45, ha="right")

    ax1 = plt.gca()

    if args.degree:
        ax2 = ax1.twinx()
        ax2.plot(range(len(sortinds)), degree_data[sortinds], marker="x", color="black")
        ax2.set_ylabel("Average Degree")

    plt.xlabel("Backend")
    plt.grid(axis="y")
    ax1.set_ylabel("Largest Connected Region")
    plt.title("CHSH Test on Each Edge to Determine Qubit Count")
    plt.savefig("count_plot_%s.pdf" % fu.timestamp_name(), bbox_inches="tight")
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
    parser.add_argument("--plot", help="Generate a plot", action="store_true")
    parser.add_argument("--degree", help="Add degree to the plot", action="store_true")
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

                err_str = "Backend %s already exists, duplicate results" % (backend)
                raise ValueError(err_str)

    if args.backends is not None:
        backends_filt = args.backends.split(",")
    else:
        backends_filt = []

    count_data = {}
    degree_data = {}

    for backend in results_dict:
        if len(backends_filt) > 0:
            if backend not in backends_filt:
                continue

        if backend == "config":
            continue

        count_data[backend] = results_dict[backend]["largest_region"]
        degree_data[backend] = results_dict[backend]["average_degree"]
        print(
            "Backend %s, Largest Connected Region: %d" % (backend, count_data[backend])
        )
        print("Backend %s, Average Degree: %f" % (backend, degree_data[backend]))

    if args.plot:
        generate_plot(count_data, degree_data, args)

    elif args.plot:
        print("Need to run mean/max also")
