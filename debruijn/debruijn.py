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

from typing import Iterator, Dict, List
import argparse
import os
import sys
import statistics
import textwrap
import matplotlib
from pathlib import Path
from operator import itemgetter
import random
random.seed(9001)
from random import randint
import matplotlib.pyplot as plt
import networkx as nx
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)

matplotlib.use("Agg")

__author__ = "Emilie-Jeanne MAYI"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Emilie-Jeanne MAYI"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Emilie-Jeanne MAYI"
__email__ = "emilie-jeanne.mayi@etu.u-paris.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments"
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    # Liste des sequences
    sequences = []
    # liste des qualites
    quality = []

    # Retourne un generateur de séquence
    with open(fastq_file, 'rt') as input_file:
        # Remplissage des listes
        for line in input_file:
            yield input_file.readline().strip() # lecture de sequence
            input_file.readline() # lecture du +
            input_file.readline() # lecture de la qualite


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    # Definition de la taille de kmer
    for index in range(len(read) - (kmer_size - 1)):
        yield read[index:(index + kmer_size)]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    # Récupération des séquences
    read = read_fastq(fastq_file)
    # Initialisation d'une liste pour récupérer l'ensemble des k-mers
    liste_kmer = []
    for sequence in read:
        # Recupération des kmer par séquence
        generator_kmer = cut_kmer(sequence, kmer_size)

        # Mise à jour de la liste de kmer
        liste_kmer = liste_kmer + list(generator_kmer)
    uniq_kmer = set(liste_kmer)
    dico_kmer = dict([[kmer, liste_kmer.count(kmer)] for kmer in uniq_kmer])
    return dico_kmer


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    # Initialisation d'un graphique vide
    G = DiGraph()
    kmer_keys = list(kmer_dict.keys())
    # Ajout des nodes
    # Parcours de la liste de k-mer
    for kmer in kmer_keys:
        # Parcours de la liste de k-mer : potentiels nodes voisins
        for other_kmer in kmer_keys:
            # Test : k-mers successifs/liés
            if kmer[1:] == other_kmer[:-1]:
                G.add_edge(u_of_edge=kmer,v_of_edge=other_kmer, weight= kmer_dict[kmer])
    return G


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if delete_entry_node and delete_sink_node:
        # Suppression de tous les nodes du chemin
        graph.remove_nodes_from(path_list)
    elif not delete_entry_node and not delete_sink_node:
        # Suppression de tous les nodes du chemin saut le premier et le dernier
        graph.remove_nodes_from(path_list[1:-1])
    else:
        if delete_entry_node:
            # Suppression de tous les nodes du chemin saut le dernier
            graph.remove_nodes_from(path_list[:-1])
        elif delete_sink_node:
            # Suppression de tous les nodes du chemin sauf le premier
            graph.remove_nodes_from(path_list[1:])
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    pass


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    pass


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    pass


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    pass


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    pass


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = []

    # Parcours les nodes du graphe
    for index_node in range(len( list(graph.nodes()) )):
        # Sommet actuellement traversé
        current_node = list(graph.nodes())[index_node]
        # Récupère la liste des prédécesseurs du node
        dict_keyiterator_0 = graph.predecessors( current_node )
        # Test : node de départ
        if list(dict_keyiterator_0) == 0:
            print(f"node sans predecesseur : {current_node}")
            starting_nodes.append( current_node )
    return starting_nodes


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sinking_nodes = []
    # Parcours les nodes du graphe
    for index_node in range(len( list(graph.nodes()) )):
        # Sommet actuellement traversé
        current_node = list(graph.nodes())[index_node]
        # Récupère la liste des prédécesseurs du node
        dict_keyiterator_0 = graph.successors(current_node)
        # Test : node de fin
        if len(list(dict_keyiterator_0)) == 0:
            print(f"Sinking node : {current_node}")
            sinking_nodes.append(current_node)
    return sinking_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    list_contig = []
    # Parcourt tous les de noeuds de départs
    for start_node in starting_nodes:
        # Parcourt touts les noeuds d'arrivée
        for end_node in ending_nodes:
            # Test : Chemin existant
            if has_path(graph, start_node, end_node):
                # Récupération de chemin sous forme de liste
                chemin = list(all_simple_paths(G=graph, source=start_node, target=end_node))
                # Reconstruction du contig (séquence)
                init_contig = chemin[0][0]
                for kmer in chemin[0][1:]:
                    init_contig = init_contig + kmer[-1]    
                # Ajout du tuple (contig, longueur)
                list_contig.append( (init_contig, len(init_contig)) )
    return list_contig


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    # Fonction save_contigs
    with open (output_file, "w") as output:
        for index_contig, contig in enumerate(contigs_list):
            output.write(f">contig_{index_contig}={contig[1]}\n{textwrap.fill(contig[0], width=80)}\n")      
    print(f"Impression du fichier: {output_file}")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Lecture du fichier + Construction du graphe
    filename = isfile(".data/eva71_hundred_reads.fq")
    dico_kmer = build_kmer_dict(fastq_file=filename, kmer_size= 5)
    graph = build_graph(dico_kmer)

    # Résolution des bulles

    # Résolution des pointes d’entrée et de sortie
    
    # Ecriture du/des contig
    # save_contigs()

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
