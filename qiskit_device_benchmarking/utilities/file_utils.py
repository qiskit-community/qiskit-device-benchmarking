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

"""File utilities for the device benchmarking."""

import yaml
import datetime


def import_yaml(fstr):
    with open(fstr, "r") as stream:
        data_imp = yaml.safe_load(stream)

    return data_imp


def timestamp_name():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")


def export_yaml(fstr, exp_data):
    with open(fstr, "w") as fout:
        yaml.dump(exp_data, fout, default_flow_style=None)
