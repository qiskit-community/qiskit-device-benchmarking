# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A module for visualizing device coupling maps"""

# A simplified alteration of the qiskit version to just use matplotlib
# Will only work for Eagle, HeronR1, HeronR2

from typing import List

import rustworkx as rx
from rustworkx.visualization import mpl_draw

from qiskit.exceptions import QiskitError
from qiskit.utils import optionals as _optionals
from qiskit.transpiler.coupling import CouplingMap


def _get_backend_interface_version(backend):
    backend_interface_version = getattr(backend, "version", None)
    return backend_interface_version


@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_gate_map(
    backend,
    figsize=None,
    plot_directed=False,
    label_qubits=True,
    qubit_size=None,
    line_width=4,
    font_size=None,
    qubit_color=None,
    qubit_labels=None,
    line_color=None,
    font_color="white",
    ax=None,
    qubit_coordinates=None,
):
    """Plots the gate map of a device.

    Args:
        backend (Backend): The backend instance that will be used to plot the device
            gate map.
        figsize (tuple): Output figure size (wxh) in inches.
        plot_directed (bool): Plot directed coupling map.
        label_qubits (bool): Label the qubits.
        qubit_size (float): Size of qubit marker.
        line_width (float): Width of lines.
        font_size (int): Font size of qubit labels.
        qubit_color (list): A list of colors for the qubits
        qubit_labels (list): A list of qubit labels
        line_color (list): A list of colors for each line from coupling_map.
        font_color (str): The font color for the qubit labels.
        ax (Axes): A Matplotlib axes instance.
        qubit_coordinates (Sequence): An optional sequence input (list or array being the
            most common) of 2d coordinates for each qubit. The length of the
            sequence much match the number of qubits on the backend. The sequence
            should be the planar coordinates in a 0-based square grid where each
            qubit is located.

    Returns:
        Figure: A Matplotlib figure instance.

    Raises:
        QiskitError: if tried to pass a simulator, or if the backend is None,
            but one of num_qubits, mpl_data, or cmap is None.
        MissingOptionalLibraryError: if matplotlib not installed.

    Example:

        .. plot::
           :include-source:

           from qiskit.providers.fake_provider import GenericBackendV2
           from qiskit.visualization import plot_gate_map

           backend = GenericBackendV2(num_qubits=5)

           plot_gate_map(backend)
    """
    qubit_coordinates_map = {}

    qubit_coordinates_map[127] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [1, 0],
        [1, 4],
        [1, 8],
        [1, 12],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [2, 11],
        [2, 12],
        [2, 13],
        [2, 14],
        [3, 2],
        [3, 6],
        [3, 10],
        [3, 14],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [4, 11],
        [4, 12],
        [4, 13],
        [4, 14],
        [5, 0],
        [5, 4],
        [5, 8],
        [5, 12],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [6, 11],
        [6, 12],
        [6, 13],
        [6, 14],
        [7, 2],
        [7, 6],
        [7, 10],
        [7, 14],
        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
        [8, 11],
        [8, 12],
        [8, 13],
        [8, 14],
        [9, 0],
        [9, 4],
        [9, 8],
        [9, 12],
        [10, 0],
        [10, 1],
        [10, 2],
        [10, 3],
        [10, 4],
        [10, 5],
        [10, 6],
        [10, 7],
        [10, 8],
        [10, 9],
        [10, 10],
        [10, 11],
        [10, 12],
        [10, 13],
        [10, 14],
        [11, 2],
        [11, 6],
        [11, 10],
        [11, 14],
        [12, 1],
        [12, 2],
        [12, 3],
        [12, 4],
        [12, 5],
        [12, 6],
        [12, 7],
        [12, 8],
        [12, 9],
        [12, 10],
        [12, 11],
        [12, 12],
        [12, 13],
        [12, 14],
    ]

    qubit_coordinates_map[133] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [0, 14],
        [1, 0],
        [1, 4],
        [1, 8],
        [1, 12],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [2, 11],
        [2, 12],
        [2, 13],
        [2, 14],
        [3, 2],
        [3, 6],
        [3, 10],
        [3, 14],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [4, 11],
        [4, 12],
        [4, 13],
        [4, 14],
        [5, 0],
        [5, 4],
        [5, 8],
        [5, 12],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [6, 11],
        [6, 12],
        [6, 13],
        [6, 14],
        [7, 2],
        [7, 6],
        [7, 10],
        [7, 14],
        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
        [8, 11],
        [8, 12],
        [8, 13],
        [8, 14],
        [9, 0],
        [9, 4],
        [9, 8],
        [9, 12],
        [10, 0],
        [10, 1],
        [10, 2],
        [10, 3],
        [10, 4],
        [10, 5],
        [10, 6],
        [10, 7],
        [10, 8],
        [10, 9],
        [10, 10],
        [10, 11],
        [10, 12],
        [10, 13],
        [10, 14],
        [11, 2],
        [11, 6],
        [11, 10],
        [11, 14],
        [12, 0],
        [12, 1],
        [12, 2],
        [12, 3],
        [12, 4],
        [12, 5],
        [12, 6],
        [12, 7],
        [12, 8],
        [12, 9],
        [12, 10],
        [12, 11],
        [12, 12],
        [12, 13],
        [12, 14],
        [13, 0],
        [13, 4],
        [13, 8],
        [13, 12],
    ]

    qubit_coordinates_map[156] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [0, 14],
        [0, 15],
        [1, 3],
        [1, 7],
        [1, 11],
        [1, 15],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [2, 11],
        [2, 12],
        [2, 13],
        [2, 14],
        [2, 15],
        [3, 1],
        [3, 5],
        [3, 9],
        [3, 13],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [4, 11],
        [4, 12],
        [4, 13],
        [4, 14],
        [4, 15],
        [5, 3],
        [5, 7],
        [5, 11],
        [5, 15],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [6, 11],
        [6, 12],
        [6, 13],
        [6, 14],
        [6, 15],
        [7, 1],
        [7, 5],
        [7, 9],
        [7, 13],
        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
        [8, 11],
        [8, 12],
        [8, 13],
        [8, 14],
        [8, 15],
        [9, 3],
        [9, 7],
        [9, 11],
        [9, 15],
        [10, 0],
        [10, 1],
        [10, 2],
        [10, 3],
        [10, 4],
        [10, 5],
        [10, 6],
        [10, 7],
        [10, 8],
        [10, 9],
        [10, 10],
        [10, 11],
        [10, 12],
        [10, 13],
        [10, 14],
        [10, 15],
        [11, 1],
        [11, 5],
        [11, 9],
        [11, 13],
        [12, 0],
        [12, 1],
        [12, 2],
        [12, 3],
        [12, 4],
        [12, 5],
        [12, 6],
        [12, 7],
        [12, 8],
        [12, 9],
        [12, 10],
        [12, 11],
        [12, 12],
        [12, 13],
        [12, 14],
        [12, 15],
        [13, 3],
        [13, 7],
        [13, 11],
        [13, 15],
        [14, 0],
        [14, 1],
        [14, 2],
        [14, 3],
        [14, 4],
        [14, 5],
        [14, 6],
        [14, 7],
        [14, 8],
        [14, 9],
        [14, 10],
        [14, 11],
        [14, 12],
        [14, 13],
        [14, 14],
        [14, 15],
    ]

    backend_version = _get_backend_interface_version(backend)
    if backend_version <= 1:
        if backend.configuration().simulator:
            raise QiskitError("Requires a device backend, not simulator.")
        config = backend.configuration()
        num_qubits = config.n_qubits
        coupling_map = CouplingMap(config.coupling_map)
        name = backend.name()
    else:
        num_qubits = backend.num_qubits
        coupling_map = backend.coupling_map
        name = backend.name
    if qubit_coordinates is None and ("ibm" in name or "fake" in name):
        qubit_coordinates = qubit_coordinates_map.get(num_qubits, None)

    if qubit_coordinates:
        if len(qubit_coordinates) != num_qubits:
            raise QiskitError(
                f"The number of specified qubit coordinates {len(qubit_coordinates)} "
                f"does not match the device number of qubits: {num_qubits}"
            )

    if qubit_coordinates is None:
        raise QiskitError(
            "No matching coordinate found, code has only 127, 133, 156 backends"
        )

    return plot_coupling_map(
        num_qubits,
        qubit_coordinates,
        coupling_map.get_edges(),
        figsize,
        plot_directed,
        label_qubits,
        qubit_size,
        line_width,
        font_size,
        qubit_color,
        qubit_labels,
        line_color,
        font_color,
        ax,
        planar=rx.is_planar(coupling_map.graph.to_undirected(multigraph=False)),
    )


@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_coupling_map(
    num_qubits: int,
    qubit_coordinates: List[List[int]],
    coupling_map: List[List[int]],
    figsize=None,
    plot_directed=False,
    label_qubits=True,
    qubit_size=None,
    line_width=4,
    font_size=None,
    qubit_color=None,
    qubit_labels=None,
    line_color=None,
    font_color="white",
    ax=None,
    *,
    planar=True,
):
    """Plots an arbitrary coupling map of qubits (embedded in a plane).

    Args:
        num_qubits (int): The number of qubits defined and plotted.
        qubit_coordinates (List[List[int]]): A list of two-element lists, with entries of each nested
            list being the planar coordinates in a 0-based square grid where each qubit is located.
        coupling_map (List[List[int]]): A list of two-element lists, with entries of each nested
            list being the qubit numbers of the bonds to be plotted.
        figsize (tuple): Output figure size (wxh) in inches.
        plot_directed (bool): Plot directed coupling map.
        label_qubits (bool): Label the qubits.
        qubit_size (float): Size of qubit marker.
        line_width (float): Width of lines.
        font_size (int): Font size of qubit labels.
        qubit_color (list): A list of colors for the qubits
        qubit_labels (list): A list of qubit labels
        line_color (list): A list of colors for each line from coupling_map.
        font_color (str): The font color for the qubit labels.
        ax (Axes): A Matplotlib axes instance.
        planar (bool): If the coupling map is planar or not. Default: ``True`` (i.e. it is planar)

    Returns:
        Figure: A Matplotlib figure instance.

    Raises:
        MissingOptionalLibraryError: If matplotlib or graphviz is not installed.
        QiskitError: If length of qubit labels does not match number of qubits.

    Example:

        .. plot::
           :include-source:

            from qiskit.visualization import plot_coupling_map

            num_qubits = 8
            qubit_coordinates = [[0, 1], [1, 1], [1, 0], [1, 2], [2, 0], [2, 2], [2, 1], [3, 1]]
            coupling_map = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 5], [5, 6], [2, 4], [6, 7]]
            plot_coupling_map(num_qubits, qubit_coordinates, coupling_map)
    """

    if qubit_size is None:
        qubit_size = 30

    if qubit_labels is None:
        qubit_labels = list(range(num_qubits))
    else:
        if len(qubit_labels) != num_qubits:
            raise QiskitError("Length of qubit labels does not equal number of qubits.")

    if not label_qubits:
        qubit_labels = [""] * num_qubits

    # set coloring
    if qubit_color is None:
        qubit_color = ["#648fff"] * num_qubits
    if line_color is None:
        line_color = ["#648fff"] * len(coupling_map)

    if num_qubits == 1:
        graph = rx.PyDiGraph()
        graph.add_node(0)
    else:
        graph = CouplingMap(coupling_map).graph

    if not plot_directed:
        line_color_map = dict(zip(graph.edge_list(), line_color))
        graph = graph.to_undirected(multigraph=False)
        line_color = [line_color_map[edge] for edge in graph.edge_list()]

    for node in graph.node_indices():
        graph[node] = node

    for edge_index in graph.edge_indices():
        graph.update_edge_by_index(edge_index, edge_index)

    if qubit_coordinates:
        qubit_coordinates = {
            i: qubit_coordinates[i][::-1] for i in range(len(qubit_coordinates))
        }

    if font_size is None:
        max_characters = max(1, max(len(str(x)) for x in qubit_labels))
        if max_characters == 1:
            font_size = 14
        elif max_characters == 2:
            font_size = 10
        elif max_characters == 3:
            font_size = 8
        else:
            font_size = 1

    def node_label(node):
        return str(qubit_labels[node])

    mpl_draw(
        graph,
        pos=qubit_coordinates,
        labels=node_label,
        font_size=font_size,
        font_color=font_color,
        with_labels=label_qubits,
        node_color=qubit_color,
        edge_color=line_color,
        ax=ax,
    )
