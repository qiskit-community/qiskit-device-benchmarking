# from qiskit-experiments-internal/qiskit_experiments_internal/tiled_experiment/partition.py
"""
Functions to partition nodes and edges into groups, such that nodes/edges
in a group are distant from each other
"""

import rustworkx as rx


def build_coupling_graph(backend, multigraph=False):
    """
    Build a rustworkx undirected graph from the backend's coupling map.

    If multigraph=True then parallel edges in the coupling map will produce parallel edges in the
    constructed graph.
    For example, if both [1, 2] and [2, 1] are in the coupling map, then the graph will contain two
    edges between the nodes labeled 1 and 2.

    The graph's edges are labeled with the coupling edge. So, the two edges from the previous
    example will be labeled [1, 2] and [2, 1].

    If multigraph=False then parallel edges will produce a single edge in the graph; the node order
    in the edge's label will be arbitrary.
    """
    graph = rx.PyGraph(multigraph=multigraph)
    graph.add_nodes_from(list(range(backend.configuration().num_qubits)))
    for coupling_edge in backend.configuration().coupling_map:
        graph.add_edge(coupling_edge[0], coupling_edge[1], coupling_edge)

    return graph


def build_line_graph(graph):
    """
    Return the line graph of the given graph.

    The line graph contains a node for each edge in the original graph.
    The node's label is the original graph's edge.
    Nodes in the line graph are connected by an (unlabeled) edge iff the original edges share a node
    in the original graph.

    Example: if the original graph has four nodes and edges [0, 1], [1, 2], [2, 3]
    then the line graph has three nodes labeled [0, 1], [1, 2], [2, 3]
    and two edges: between the nodes labeled [0, 1] and the node labeled [1, 2]; and between the
    node labeled [1, 2] and the node labeled [2, 3].
    """
    line_graph = rx.PyGraph()

    for edge in graph.edges():
        line_graph_node = line_graph.add_node(edge)

        for existing_line_graph_node in range(line_graph_node):
            existing_edge = line_graph[existing_line_graph_node]
            if set(edge).intersection(set(existing_edge)):
                line_graph.add_edge(line_graph_node, existing_line_graph_node, None)

    return line_graph


def build_distance_graph(graph, distance, node_subset=None):
    """
    The nodes of the distance graph are the same as in the original graph.
    Two nodes are connected by an edge if the distance between them
    in the original graph is smaller than `distance`.
    The nodes of the distance graph are labeled as in the original graphs.
    The edges are not labeled.

    If `node_subset` is not None then only the nodes specified in `node_subset`
    will also become nodes of the distance graph.
    """
    distance_matrix = rx.distance_matrix(graph)
    dist_graph = rx.PyGraph()
    map_node_labels_to_dist_matrix_indices = {}  # pylint: disable=invalid-name

    if node_subset is None:
        node_subset = graph.nodes()

    for dist_matrix_index1, node_label1 in enumerate(graph.nodes()):
        if node_label1 in node_subset:
            dist_graph_index1 = dist_graph.add_node(node_label1)
            map_node_labels_to_dist_matrix_indices[str(node_label1)] = dist_matrix_index1
            for dist_graph_index2 in range(dist_graph_index1):
                node_label2 = dist_graph[dist_graph_index2]
                dist_matrix_index2 = map_node_labels_to_dist_matrix_indices[str(node_label2)]
                if distance_matrix[dist_matrix_index1, dist_matrix_index2] < distance:
                    dist_graph.add_edge(dist_graph_index1, dist_graph_index2, None)

    return dist_graph


def partition_qubits(backend, distance, node_subset=None):
    """
    Partitions the nodes of the backend's coupling graph.
    See details in the documentation of partition_nodes.
    """
    graph = build_coupling_graph(backend)
    return partition_nodes(graph, distance, node_subset)


def partition_qubit_pairs(backend, distance, multigraph=False, edge_subset=None, node_subset=None):
    """
    Partitions the edges of the backend's coupling graph.
    The `multigraph` parameter is for the graph construction,
    see details in the documentation of `build_coupling_graph`.
    For the partitioning, see the documentation of `partition_edges`.
    """
    graph = build_coupling_graph(backend, multigraph)
    return partition_edges(graph, distance, edge_subset, node_subset)


def partition_nodes(graph, distance, node_subset=None):
    """
    Partitions the graph nodes into groups, such that in each group the
    minimum distance (number of edges in the shortest path) between nodes is at least `distance`.

    Returns a list of list of list of integers, with the following explanation:
    - Each node label is encapsulated in a singleton list. This is in order to be consistent with
      groups of multiple nodes, like groups of edges.
    - Each group of nodes constitutes a list.
    - We return the list of groups. For example: [[[0], [2]], [[1]]] means two groups, one of nodes
      0 and 2 and the other group contains only node 1.

    If `node_subset` is not None then the partitioning contains only nodes that belong to `node_subset`.
    """

    # A very naive algorithm, which  was chosen (at least for now) because it's easy
    # to implement. I didn't check if it has any guarantees about performance, number of node
    # groups, or equitability (if we want the groups to be more-or-less of
    # equal size). The literature is filled with algorithms for node colorings,
    # including for the case of the distance>2 constraint. In particular,
    # we don't exploit information that we may have about the structure of the graph, like
    # an upper-bounded degree.

    # Construct the distance graph: an edge between two nodes
    # if the distance between them is smaller than `distance`
    dist_graph = build_distance_graph(graph, distance, node_subset)

    # Color the distance graph: nodes that are adjacent in the distance graph
    # will be assigned different colors
    colors = rx.graph_greedy_color(dist_graph)

    # Partition the nodes according to their colors
    return [
        [[dist_graph[node]] for node in dist_graph.node_indices() if colors[node] == c]
        for c in set(colors.values())
    ]


def partition_edges(graph, distance, edge_subset=None, node_subset=None):
    """
    Partitions the edges in the graph into groups, such that in each group the
    minimum distance (one plus number of edges in the shortest path) between edges is at least
    `distance`.

    Returns a list of list of pairs of integers, i.e., a list of groups of edges.

    If `node_subset` is not None then the partitioning contains only edges where both nodes belong to
    `node_subset`.
    If `edge_subset` is not None then the partitioning contains only edges that belong to `edge_subset`.
    It is not allowed to set both `node_subset` and `edge_subset` to values different from None.
    """

    # Algorithm:
    # - We construct the line graph of the input graph. In the line graph,
    #   every node represents an edge of the original graph. Nodes in the line graph
    #   are connected by an edge if the respective edges in the original graph share at least
    #   one node.
    # - We construct the distance graph of the line graph. The nodes of the distance graph are the
    #   same as in the line graph. Two nodes are connected by an edge if the distance between them
    #   in the line graph is smaller than `distance`.
    # - We color the nodes of the distance graph (coloring a graph means that adjacent
    #   nodes are assigned different colors). This induces a coloring to the edges of the
    #   original graph that satisfies the distance constraint.
    #
    # This is a very naive algorithm, and it was chosen (at least for now) because it's easy
    # to implement. I didn't check if it has any guarantees about performance, number of edge
    # groups, or equitability (if we want the groups to be more-or-less of
    # equal size). The literature is filled with algorithms for node and edge colorings,
    # including for the case of the distance>2 constraint. In particular,
    # we don't exploit information that we may have about the structure of the graph, like
    # an upper-bounded degree.

    if edge_subset is None:
        if node_subset is None:
            edge_subset = graph.edges()
        else:
            edge_subset = [
                edge for edge in graph.edges() if edge[0] in node_subset and edge[1] in node_subset
            ]
    elif node_subset is not None:
        raise ValueError("Impossible to set both edge_subset and node_subset")

    line_graph = build_line_graph(graph)

    # Construct the distance graph
    dist_graph = build_distance_graph(line_graph, distance, edge_subset)

    # Color the distance graph
    colors = rx.graph_greedy_color(dist_graph)

    return [
        [dist_graph[dist_node] for dist_node in dist_graph.node_indices() if colors[dist_node] == c]
        for c in set(colors.values())
    ]
