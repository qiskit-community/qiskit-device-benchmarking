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

"""Graph utilities for the device benchmarking."""

import rustworkx as rx
import numpy as np
import copy


def paths_flatten(paths):
    """Flatten a list of paths from retworkx

    Args:
        paths: all_pairs_all_simple_paths

    Returns:
        flat list of lists of qubit chains
    """
    return [list(val) for ps in paths.values() for vals in ps.values() for val in vals]


def remove_permutations(paths):
    """remove permutations from the paths

    Args:
        paths: list of qubit chains

    Returns:
        list of qubit chains without permutations
    """

    new_path = []
    for path_i in paths:
        # check already in the new_path
        if path_i in new_path:
            continue

        # reverse and check
        path_i.reverse()
        if path_i in new_path:
            continue
        path_i.reverse()

        new_path.append(path_i)

    return new_path


def path_to_edges(paths, coupling_map=None):
    """Converse a list of paths into a list of edges that are in the
    coupling_map if defined

    If already edges (length 2 path) then convert into the edge that's in the
    coupling map

    Args:
        paths: list of qubit chains

    Returns:
        list of qubit paths in terms of the edges to traverse.
    """

    new_path = []
    for path_i in paths:
        if len(path_i) > 2:
            new_path.append([])

        for i in range(len(path_i) - 1):
            tmp_set = path_i[i : (i + 2)]
            if coupling_map is not None:
                if tuple(tmp_set) not in coupling_map and tmp_set not in coupling_map:
                    tmp_set.reverse()
                    if (
                        tuple(tmp_set) not in coupling_map
                        and tmp_set not in coupling_map
                    ):
                        raise ValueError("Path not found in coupling map")

            if len(path_i) > 2:
                new_path[-1].append(tmp_set)
            else:
                new_path.append(tmp_set)

    return new_path


def build_sys_graph(nq, coupling_map, faulty_qubits=None):
    """Build a system graph

    Args:
        nq: number of qubits
        coupling_map: coupling map in list form
        faulty_qubits: list of faulty qubits (will remove from graph)

    Returns:
        undirected graph with no duplicate edges
    """

    if faulty_qubits is not None:
        coupling_map2 = []

        for i in coupling_map:
            if (i[0] not in faulty_qubits) and (i[1] not in faulty_qubits):
                coupling_map2.append(i)

        coupling_map = coupling_map2

    G = rx.PyDiGraph()
    G.add_nodes_from(range(nq))
    G.add_edges_from_no_data([tuple(x) for x in coupling_map])
    return G.to_undirected(multigraph=False)


def get_iso_qubit_list(G):
    """Return a set of lists of isolated (separated by at least one idle qubit)
    qubits using graph coloring

    Args:
        G: system graph (assume G.to_undirected(multigraph=False) has been run)

    Returns:
        list of qubit lists
    """

    qlists = {}
    node_dict = rx.graph_greedy_color(G)
    for i in node_dict:
        if node_dict[i] in qlists:
            qlists[node_dict[i]].append(i)
        else:
            qlists[node_dict[i]] = [i]

    qlists = list(qlists.values())
    for i in range(len(qlists)):
        qlists[i] = list(np.sort(qlists[i]))

    return qlists


def get_disjoint_edge_list(G):
    """Return a set of disjoint edges using graph coloring

    Args:
        G: system graph (assume G.to_undirected(multigraph=False) has been run)

    Returns:
        list of list of edges
    """

    edge_lists = {}
    edge_dict = rx.graph_greedy_edge_color(G)
    for i in edge_dict:
        if edge_dict[i] in edge_lists:
            edge_lists[edge_dict[i]].append(G.edge_list()[i])
        else:
            edge_lists[edge_dict[i]] = [G.edge_list()[i]]

    return list(edge_lists.values())


def get_separated_sets(G, node_sets, min_sep=1, nsets=-1):
    """Given a list node sets separate out into lists where
    the sets in each list are separated by min_sep

    This could be quite slow!

    Args:
        G: system graph
        node_sets: list of list of nodes
        min_sep: minimum separation between node sets
        nsets: number of sets to truncate at, if -1 then make all sets

    Returns:
        list of list of list of nodes each separated by min_sep
    """

    node_sets_sep = [[]]
    cur_ind1 = 0
    cur_ind2 = 0

    node_sets_tmp = copy.deepcopy(node_sets)

    # get all node to node distances in a dictionary
    all_dists = rx.all_pairs_dijkstra_path_lengths(G, lambda a: 1)

    while len(node_sets_tmp) > 0:
        if cur_ind2 >= len(node_sets_tmp):
            if nsets > 0 and (cur_ind1 + 2) > nsets:
                break

            node_sets_sep.append([])
            cur_ind1 += 1
            cur_ind2 = 0

        add_set = True
        for node_set in node_sets_sep[cur_ind1]:
            if not sets_min_dist(all_dists, node_set, node_sets_tmp[cur_ind2], min_sep):
                add_set = False
                cur_ind2 += 1
                break

        if add_set:
            node_sets_sep[cur_ind1].append(node_sets_tmp[cur_ind2])
            node_sets_tmp.pop(cur_ind2)

    return node_sets_sep


def sets_min_dist(dist_dict, set1, set2, min_sep):
    """Calculate if two sets are min_sep apart

    Args:
        dist_dict: dictionary of distances between nodes
        set1,2: the two sets
        min_sep: minimum separation

    Returns:
        True/False
    """

    # dummy check
    if set(set1) & set(set2):
        return False

    for i in set1:
        for j in set2:
            if dist_dict[i][j] < min_sep:
                return False

    return True


def create_graph_dict(coupling_map: list, nq: int) -> dict:
    graph_dict = {i: [] for i in range(nq)}

    for edge in coupling_map:
        if edge[1] not in graph_dict[edge[0]]:
            graph_dict[edge[0]].append(edge[1])

        if edge[0] not in graph_dict[edge[1]]:
            graph_dict[edge[1]].append(edge[0])

    return graph_dict


def iter_neighbors(
    graph_dict: dict,
    cur_node: int,
    err_map: dict,
    best_fid: list,
    fid_cutoff: float,
    cur_list: list,
    chain_fid: float,
    pathlen: int,
) -> list:
    """
    takes a list of paths through a graph and adds to
    it all the neighbor qubits of the last point as long
    as the graph does fold on itself. This version is different than the above
    in that it tracks a best fidelity and will skip paths
    that don't seem viable

    if the lists get long enough return the lists

    Args:
        graph_dict: dictionary of nodes and their neighbors
        cur_node: current node on the graph
        err_map: map of edge errors (AVERAGE gate error)
        best_fid: list of length 1 (so mutable) of the best fidelity
        fid_cutoff: the percentage (0->1) of the best fidelity at that chain length
        to cutoff the search
        cur_list: current path through graph
        chain_fid: fidelit of the current path
        pathlen: length of the path we are trying to find

    Returns:
        new_list: a list of all the paths appended to cur_list
    """

    new_list = []
    for i in graph_dict[cur_node]:
        # no backtracking
        if len(cur_list) > 1 and i in cur_list:
            continue

        if "%d_%d" % (cur_node, i) in err_map:
            edge_err = err_map["%d_%d" % (cur_node, i)]
        else:
            edge_err = err_map["%d_%d" % (i, cur_node)]

        # if the edge does not seem viable skip
        new_fid = chain_fid * (1 - 5 / 4 * edge_err)
        if new_fid < (fid_cutoff * best_fid[0]) ** ((len(cur_list) + 1) / pathlen):
            continue
        # add the current node to the list
        cur_list_tmp = cur_list.copy()
        cur_list_tmp.append(i)

        # check if the list is long enough
        if len(cur_list_tmp) < pathlen:
            # if not then continue to add to it
            tmp_new_list = iter_neighbors(
                graph_dict,
                i,
                err_map,
                best_fid,
                fid_cutoff,
                cur_list_tmp,
                new_fid,
                pathlen,
            )
            for tmp_node in tmp_new_list:
                if len(tmp_node) != 0:
                    new_list.append(tmp_node)
        else:
            # append the path to the list
            if new_fid > best_fid[0]:
                best_fid[0] = new_fid
            new_list.append(cur_list_tmp)
    return new_list
