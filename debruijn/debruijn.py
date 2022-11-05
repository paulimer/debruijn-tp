#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import itertools
import os
import sys
import networkx as nx
import matplotlib
from operator import itemgetter
import random
random.seed(9001)
from random import randint
import statistics
import matplotlib.pyplot as plt
matplotlib.use("Agg")

__author__ = "Paul Etheimer"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Paul Etheimer"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Paul Etheimer"
__email__ = "your@email.fr"
__status__ = "Developpement"

def isfile(path):
    """Check if path is an existing file.
      :Parameters:
          path: Path to the file
    """
    if not os.path.isfile(path):
        if os.path.isdir(path):
            msg = "{0} is a directory".format(path)
        else:
            msg = "{0} does not exist.".format(path)
        raise argparse.ArgumentTypeError(msg)
    return path


def get_arguments():
    """Retrieves the arguments of the program.
      Returns: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=
                                     "{0} -h"
                                     .format(sys.argv[0]))
    parser.add_argument('-i', dest='fastq_file', type=isfile,
                        required=True, help="Fastq file")
    parser.add_argument('-k', dest='kmer_size', type=int,
                        default=22, help="k-mer size (default 22)")
    parser.add_argument('-o', dest='output_file', type=str,
                        default=os.curdir + os.sep + "contigs.fasta",
                        help="Output contigs in fasta file")
    parser.add_argument('-f', dest='graphimg_file', type=str,
                        help="Save graph as image (png)")
    return parser.parse_args()


def read_fastq(fastq_file):
    """
    Reads a fastq file. Returns a sequence generator.

    Parameters
    ----------
    fastq_file: str
    a fastq file

    Returns
    -------
    A sequence generator
    """
    with open(fastq_file, "r") as filein:
        id = filein.readline().rstrip()
        while id:
            yield filein.readline().rstrip()
            filein.readline()
            filein.readline()
            id = filein.readline().rstrip()




def cut_kmer(read, kmer_size):
    """
    Returns a generator of kmer from a read.

    Parameters
    ----------
    read: str
    the read to take kmers from
    kmer_size: int
    the size of kmers

    Returns
    -------
    A generator of kmers
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:kmer_size+i]


def build_kmer_dict(fastq_file, kmer_size):
    """
    Creates the kmer dictionary.

    Parameters
    ----------
    fastq_file: str
    The path to the file containing the sequences
    kmer_size: int
    The size of the kmers

    Returns
    -------
    The dictionary of the kmers and their occurence
    """
    kmer_dict = {}
    for seq in read_fastq(fastq_file):
        for kmer in cut_kmer(seq, kmer_size):
            kmer_dict[kmer] = kmer_dict.get(kmer, 0) + 1
    return kmer_dict


def build_graph(kmer_dict):
    """
    Creates the de Bruijn graph from a kmer dictionary

    Parameters
    ----------
    kmer_dict: dict
    the kmer dictionary

    Returns
    -------
    A graph of prefixes and suffixes of those kmers.
    """
    res_graph = nx.DiGraph()
    for kmer, value in kmer_dict.items():
        res_graph.add_edge(kmer[:-1], kmer[1:], weight=value)
    return res_graph


def remove_paths(graph, paths, delete_entry_node, delete_sink_node):
    """
    Removes nodes along a path of a given graph.

    Parameters
    ----------
    graph: networkx.DiGraph
    The input graph
    delete_entry_node: bool
    whether to delete the first node of the path
    delete_sink_node: bool
    whether to delete the last node of the path
    """
    for path in paths:
        if not delete_entry_node:
            path = path[1:]
        if not delete_sink_node:
            path = path[:-1]
        graph.remove_nodes_from(path)
    return graph


def select_best_path(graph, path_list, path_length, weight_avg_list,
                     delete_entry_node=False, delete_sink_node=False):
    """
    Compares and selects the best path among many in a graph. Deletes the others.

    Parameters
    ----------
    graph: networkx.DiGraph
    The input graph
    path_list: list
    A list of paths from which to select
    path_length: list
    A list of the length of the paths
    weight_avg_list: list
    A list of the average weight of the path
    delete_entry_node: bool
    whether to delete the first node of the path
    delete_sink_node: bool
    whether to delete the last node of the path

    Returns
    -------
    The graph without the worse paths.
    """
    if statistics.stdev(weight_avg_list) != 0:
        del path_list[weight_avg_list.index(max(weight_avg_list))]
    elif statistics.stdev(path_length) != 0:
        del path_list[path_length.index(max(path_length))]
    else:
        del path_list[randint(0, len(path_list) - 1)]
    graph = remove_paths(graph, path_list, delete_entry_node, delete_sink_node)
    return graph

def path_average_weight(graph, path):
    """Compute the weight of a path"""
    return statistics.mean([d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)])

def solve_bubble(graph, ancestor_node, descendant_node):
    """
    Removes all but the best path from a bubble.

    Parameters
    ----------
    graph: networkx.DiGraph
    The input graph
    ancestor_node: node
    The starting node of the bubble
    descendant_node: node
    The ending node of the bubble

    Returns
    -------
    The graph with the bubble removed.
    """
    bubble_paths = list(nx.all_simple_paths(graph, ancestor_node, descendant_node))
    path_length = [len(p) for p in bubble_paths]
    weight_avg_list = [path_average_weight(graph, p) for p in bubble_paths]
    graph = select_best_path(graph, bubble_paths, path_length, weight_avg_list)
    return graph

def simplify_bubbles(graph):
    """
    Returns a graph without bubbles.

    Parameters
    ----------
    graph: networkx.DiGraph
    The input graph

    Returns
    -------
    The graph without bubbles
    """
    nodes = list(graph.nodes())
    i = len(nodes) - 1
    while i >= 0:
        cur_node = nodes[i]
        prec_nodes = list(graph.predecessors(cur_node))
        if len(prec_nodes) > 1:
            ancestors = []
            for pred_1, pred_2 in itertools.combinations(prec_nodes, 2):
                ancestor = nx.lowest_common_ancestor(graph, pred_1, pred_2)
                if ancestor:
                    ancestors += [ancestor]
            ancestor_indexes = [nodes.index(anc) for anc in ancestors]
            graph = solve_bubble(graph, nodes[min(ancestor_indexes)], cur_node)
            i = min(ancestor_indexes)
        else:
            i -= 1
    return graph

            


def solve_entry_tips(graph, starting_nodes):
    """
    Simplifies the entry of the graph.

    Parameters
    ----------
    graph: networkx.DiGraph
    The input graph
    starting_nodes: list
    a list of the entry nodes

    Returns
    -------
    The graog without the entry tips
    """
    if len(starting_nodes) == 1:
        return graph
    in_nodes_tupples_list = list(itertools.combinations(starting_nodes, 2))
    for (node1, node2), desc in nx.all_pairs_lowest_common_ancestor(graph.reverse(), pairs=in_nodes_tupples_list):
        path_list = list(nx.all_simple_paths(graph, node1, desc)) + list(nx.all_simple_paths(graph, node2, desc))
        weights = [path_average_weight(graph, p) for p in path_list]
        lengths = [len(p) for p in path_list]
        graph = select_best_path(graph, path_list, lengths, weights, True, False)
    return graph


def solve_out_tips(graph, ending_nodes):
    """
    Simplifies the ending of the graph.

    Parameters
    ----------
    graph: networkx.DiGraph
    The input graph
    starting_nodes: list
    a list of the exit nodes

    Returns
    -------
    The graog without the out tips
    """
    if len(ending_nodes) == 1:
        return graph
    out_nodes_tupples_list = list(itertools.combinations(ending_nodes, 2))
    for (node1, node2), pred in nx.all_pairs_lowest_common_ancestor(graph, pairs=out_nodes_tupples_list):
        path_list = list(nx.all_simple_paths(graph, pred, node1)) + list(nx.all_simple_paths(graph, pred, node2))
        weights = [path_average_weight(graph, p) for p in path_list]
        lengths = [len(p) for p in path_list]
        graph = select_best_path(graph, path_list, lengths, weights, False, True)
    return graph



def get_starting_nodes(graph):
    """
    Gets the starting nodes from a input graph.

    Parameters
    ----------
    graph: networkx.DiGraph
    The input graph

    Returns
    -------
    A list of starting nodes
    """
    return [node for node in nx.nodes(graph) if len(list(graph.predecessors(node))) == 0]

def get_sink_nodes(graph):
    """
    Gets the sink nodes from a input graph.

    Parameters
    ----------
    graph: networkx.DiGraph
    The input graph

    Returns
    -------
    A list of end nodes
    """
    return [node for node in nx.nodes(graph) if len(list(graph.successors(node))) == 0]

def get_contigs(graph, starting_nodes, ending_nodes):
    """
    Finds contigs given a graph, starting nodes and ending nodes.

    Parameters
    ----------
    graph: nx.DiGraph
    The entry graph
    starting_nodes: list
    a list of starting nodes
    ending_nodes: list
    a list of ending nodes

    Returns
    -------
    A list of tuples of (contigs, contig_length).
    """
    contigs_list = []
    for sn in starting_nodes:
        for en in ending_nodes:
            if nx.has_path(graph, sn, en):
                for path in nx.all_simple_paths(graph, sn, en):
                    contig_str = path[0]
                    for node in path[1:]:
                        contig_str += node[-1]
                    contigs_list.append((contig_str, len(contig_str)))
    return contigs_list


def save_contigs(contigs_list, output_file):
    """
    Saves a contig list to an output file.

    Parameters
    ----------
    contigs_list: list
    A list of all contigs and their length (tuples)
    output_file: str
    Path to an output file
    """
    with open(output_file, "w") as fileout:
        for i, contig_tu in enumerate(contigs_list):
            fileout.write(f">contig_{i} len={contig_tu[1]}" + "\n")
            fileout.write(fill(contig_tu[0]) + "\n")


def fill(text, width=80):
    """Split text with a line return to respect fasta format"""
    return os.linesep.join(text[i:i+width] for i in range(0, len(text), width))

def draw_graph(graph, graphimg_file):
    """Draw the graph
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > 3]
    #print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] <= 3]
    #print(elarge)
    # Draw the graph with networkx
    #pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(graph, pos, edgelist=esmall, width=6, alpha=0.5,
                           edge_color='b', style='dashed')
    #nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file)


def save_graph(graph, graph_file):
    """Save the graph with pickle
    """
    with open(graph_file, "wt") as save:
            pickle.dump(graph, save)


#==============================================================
# Main program
#==============================================================
if __name__ == '__main__':
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(kmer_dict)
    graph = simplify_bubbles(graph)
    graph = solve_entry_tips(graph, get_starting_nodes(graph))
    graph = solve_out_tips(graph, get_sink_nodes(graph))
    contigs = get_contigs(graph, get_starting_nodes(graph), get_sink_nodes(graph))
    if(args.output_file):
        save_contigs(contigs, args.output_file)
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)

