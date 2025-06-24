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
Generate circuits for fast benchmark
"""

import argparse
from qiskit.transpiler import Target, CouplingMap
from qiskit import qpy

from qiskit_device_benchmarking.bench_code.mrb import MirrorQuantumVolume


def gen_bench_circuits(depths, he, output, opt_level, ntrials, twoqgate):
    """Pregenerate and transpile circuits for fast_bench
    Will generate the circuits as identity mirrors. Pauli's at
    front and back added when running

    Args:
        depths: the depths to generate for
        he: hardware efficient
        output: root file name
        opt_level: optimization level of the transpilation
        ntrials: number of circuits to generate
        twoqgate: two qubit gate

    Returns:
        None
    """

    print(depths)
    print(he)
    print(output)
    print(opt_level)
    print(ntrials)
    print(twoqgate)

    for depth in depths:
        print("Generating Depth %d Circuits" % (depth))

        # Construct mirror QV circuits on each parallel set

        # generate the circuits
        mqv_exp = MirrorQuantumVolume(
            qubits=list(range(depth)),
            trials=ntrials,
            split_inverse=True,
            pauli_randomize=False,
            middle_pauli_randomize=False,
            calc_probabilities=False,
            he=he,
        )

        # Do this so it won't compile outside the qubit sets
        cust_map = [[i, i + 1] for i in range(depth - 1)]

        cust_target = Target.from_configuration(
            basis_gates=["rz", "sx", "x", "id", twoqgate],
            num_qubits=depth,
            coupling_map=CouplingMap(cust_map),
        )

        mqv_exp.set_transpile_options(target=cust_target, optimization_level=opt_level)
        circs = mqv_exp._transpiled_circuits()

        ngates = 0
        ngates_sing = 0
        for circ in circs:
            gate_count = circ.count_ops()
            ngates += gate_count[twoqgate]
            for i in gate_count:
                if i != twoqgate and i != "rz":
                    ngates_sing += gate_count[i]

        print(
            "Total number of 2Q gates per circuit average: %f" % (ngates / len(circs))
        )
        print(
            "Total number of 1Q gates (no RZ) per circuit average: %f"
            % (ngates_sing / len(circs))
        )

        with open("%s_%d.qpy" % (output, depth), "wb") as fd:
            qpy.dump(circs, fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate circuits"
        + "for fast benchmark and save to qpy"
        + " optimized on a line"
    )
    parser.add_argument("-d", "--depths", help="depths to generate as a list")
    parser.add_argument("--he", help="Hardware efficient", action="store_true")
    parser.add_argument("-o", "--output", help="Output filename")
    parser.add_argument("-ol", "--opt_level", help="Optimization Level", default=3)
    parser.add_argument("-n", "--ntrials", help="Number of circuits", default=10)
    parser.add_argument("-g", "--twoqgate", help="Two qubit gate", default="cz")
    args = parser.parse_args()

    gen_bench_circuits(
        [int(i) for i in args.depths.split(",")],
        args.he,
        args.output,
        int(args.opt_level),
        int(args.ntrials),
        args.twoqgate,
    )
