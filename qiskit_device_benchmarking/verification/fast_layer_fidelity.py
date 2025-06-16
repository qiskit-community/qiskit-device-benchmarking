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
from typing import List
import datetime
import os

from qiskit_ibm_runtime import QiskitRuntimeService

import qiskit_device_benchmarking.utilities.layer_fidelity_utils as lfu
import qiskit_device_benchmarking.utilities.file_utils as fu


def run_fast_lf(
    backends: List[str],
    nseeds: int,
    seed: int,
    cliff_lengths: List[int],
    nshots: int,
    act_name: str,
    hgp: str,
):
    # Make a general experiment folder
    parent_path = os.path.join(os.getcwd(), "layer_fidelity")
    try:
        print(f"Creating folder {parent_path}")
        os.mkdir(parent_path)
    except FileExistsError:
        pass
    print(f"Changing directory to {parent_path}")
    os.chdir(parent_path)

    # Load the service
    print("Loading service")
    service = QiskitRuntimeService(name=act_name)

    for backend_name in backends:
        # Make an experiment folder each backend
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        directory = f"{time}_{backend_name}_layer_fidelity"
        path = os.path.join(parent_path, directory)
        print(f"Creating folder {path}")
        os.mkdir(path)
        print(f"Changing directory to {path}")
        os.chdir(path)

        # Get the real backend
        print(f"Getting backend {backend_name}")
        backend = service.backend(backend_name, instance=hgp)

        # Get 100Q chain from qiskit
        qchain = lfu.get_lf_chain(backend, 100)
        print(f"100Q chain for {backend_name} is: ", qchain)

        # Run LF
        print(f"Running LF on {backend_name}")
        exp_data = lfu.run_lf_chain(
            chain=qchain,
            backend=backend,
            nseeds=nseeds,
            seed=seed,
            cliff_lengths=cliff_lengths,
            nshots=nshots,
        )

        # Fit 2Q experiment data
        print(f"Retrieving experiment results from {backend_name}")
        exp_data.block_for_results()

        # Get LF and EPLG data per length
        results_per_length = lfu.reconstruct_lf_per_length(exp_data, qchain, backend)
        results_per_length.to_csv(
            f"{backend_name}_lf_eplg_data.csv", float_format="%.15f"
        )

        # Retrieve raw and fitted RB data
        rb_data_df = lfu.get_rb_data(exp_data)
        print(f"Saving 2Q data from {backend_name}")
        rb_data_df.to_csv(f"{backend_name}_full_rb_data.csv", float_format="%.15f")

        # Plot LF and EPLG data
        print(f"Making plots for {backend_name}")
        lfu.make_lf_eplg_plots(
            backend=backend, exp_data=exp_data, chain=qchain, machine=backend_name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fast layer fidelity "
        + "on reported qikist chain. Specify a config "
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
    parser.add_argument("--hgp", help="specify hgp / qiskit instance")
    parser.add_argument("--name", help="Account name", default="")
    parser.add_argument("--nseeds", help="number of seeds", default=6)
    parser.add_argument("--seed", help="seed to use", default=42)
    parser.add_argument("--nshots", help="number of shots", default=200)
    parser.add_argument(
        "--cliff_lengths",
        help="list of clifford lenghts [...]",
        default=[1, 10, 20, 30, 40, 60, 80, 100, 150, 200, 400],
    )
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
    # set default values unless otherwise instructed on config_dict
    if "nseeds" in config_dict.keys():
        nseeds = config_dict["nseeds"]
    else:
        nseeds = args.nseeds
    if "seed" in config_dict.keys():
        seed = config_dict["seed"]
    else:
        seed = args.seed
    if "cliff_lengths" in config_dict.keys():
        cliff_lengths = config_dict["cliff_lengths"]
    else:
        cliff_lengths = args.cliff_lengths
    if "nshots" in config_dict.keys():
        nshots = config_dict["nshots"]
    else:
        nshots = args.nshots
    if "act_name" in config_dict.keys():
        act_name = config_dict["act_name"]
    else:
        act_name = args.name

    # Run fast layer fidelity on the list of backends
    print("Running fast layer fidelity on backend(s)")
    run_fast_lf(backends, nseeds, seed, cliff_lengths, nshots, act_name, hgp)
