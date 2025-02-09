"""
Visualization utilities for knowledge graph and embeddings.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import umap
from sklearn.cluster import HDBSCAN
from collections import defaultdict

def create_knowledge_graph_visualization(
    embeddings: np.ndarray,
    titles: List[str],
    relationships: List[Tuple[str, str, str]],
    queries: Optional[List[Tuple[str, np.ndarray]]] = None,
    min_cluster_size: int = 5,
    figsize: Tuple[int, int] = (20, 20)
) -> None:
    """
    Create and save a visualization of the knowledge graph with:
    - Topic clusters from embeddings
    - Graph relationships
    - Query points (optional)

    Args:
        embeddings: Document embeddings array
        titles: List of document titles
        relationships: List of (source, target, type) relationships
        queries: Optional list of (query_text, query_embedding) tuples
        min_cluster_size: Minimum size for HDBSCAN clusters
        figsize: Figure size tuple
    """
    # Create title set for quick lookup
    title_set = set(titles)

    # Filter relationships to only include processed titles
    filtered_relationships = [
        (source, target, rel_type)
        for source, target, rel_type in relationships
        if source in title_set and target in title_set
    ]

    # Reduce dimensionality for visualization
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        metric='cosine',
        random_state=42
    )

    # Include query embeddings in dimensionality reduction if provided
    if queries:
        query_embeddings = np.array([q[1] for q in queries])
        all_embeddings = np.vstack([embeddings, query_embeddings])
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        doc_points = reduced_embeddings[:len(embeddings)]
        query_points = reduced_embeddings[len(embeddings):]
    else:
        reduced_embeddings = reducer.fit_transform(embeddings)
        doc_points = reduced_embeddings
        query_points = None

    # Cluster the documents
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_epsilon=0.1
    )
    cluster_labels = clusterer.fit_predict(doc_points)

    # Create networkx graph
    G = nx.Graph()

    # Add nodes with positions
    pos = {}
    node_colors = []
    node_sizes = []
    labels = {}

    # Add document nodes
    unique_clusters = sorted(set(cluster_labels))
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    cluster_color_map = dict(zip(unique_clusters, cluster_colors))

    # Create cluster centers for labels
    cluster_centers = defaultdict(lambda: {'x': [], 'y': [], 'titles': []})

    for i, (point, title, cluster) in enumerate(zip(doc_points, titles, cluster_labels)):
        G.add_node(title)
        pos[title] = point
        node_colors.append(cluster_color_map[cluster])
        node_sizes.append(100)  # Base size

        # Store points for cluster centers
        cluster_centers[cluster]['x'].append(point[0])
        cluster_centers[cluster]['y'].append(point[1])
        cluster_centers[cluster]['titles'].append(title)

    # Add relationships and update node sizes
    edge_colors = []
    for source, target, rel_type in filtered_relationships:
        G.add_edge(source, target)
        edge_colors.append('#666666')  # Darker gray for edges

    # Update node sizes based on degree
    for title in G.nodes():
        degree = G.degree(title)
        labels[title] = title if degree > 2 else ""
        # Find the index of this title in our original list
        try:
            idx = titles.index(title)
            node_sizes[idx] = 100 + degree * 20
        except ValueError:
            continue

    # Create figure
    plt.figure(figsize=figsize)

    # Draw edges first with increased width and opacity
    nx.draw_networkx_edges(
        G,
        pos=pos,
        edge_color=edge_colors,
        alpha=0.4,  # Increased opacity
        width=1.0   # Increased width
    )

    # Create legend handles for clusters
    legend_elements = []

    # Draw nodes and add to legend
    for cluster in unique_clusters:
        if cluster == -1:
            label = "Unclustered"
        else:
            # Get most common words from cluster titles
            titles = cluster_centers[cluster]['titles']
            # Use first title as representative
            label = f"Cluster {cluster}: {titles[0]}"

        # Add to legend
        legend_elements.append(plt.scatter(
            [],
            [],
            c=[cluster_color_map[cluster]],
            alpha=0.7,
            s=100,
            label=label
        ))

        # Add cluster label at center
        if cluster != -1:  # Don't label noise cluster
            center_x = np.mean(cluster_centers[cluster]['x'])
            center_y = np.mean(cluster_centers[cluster]['y'])
            plt.annotate(
                f'Cluster {cluster}',
                (center_x, center_y),
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='none',
                    boxstyle='round,pad=0.5'
                ),
                fontsize=10,
                ha='center',
                va='center'
            )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.7
    )

    # Add labels for high-degree nodes
    nx.draw_networkx_labels(
        G,
        pos=pos,
        labels=labels,
        font_size=8
    )

    # Add query points if provided
    if queries and query_points is not None:
        query_x = query_points[:, 0]
        query_y = query_points[:, 1]
        query_scatter = plt.scatter(
            query_x,
            query_y,
            c='red',
            marker='*',
            s=200,
            label='Queries',
            zorder=100  # Ensure queries are drawn on top
        )
        legend_elements.append(query_scatter)

        # Add query labels
        for i, (query_text, _) in enumerate(queries):
            plt.annotate(
                query_text,
                (query_x[i], query_y[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='red',
                zorder=100
            )

    plt.title("Wikipedia Knowledge Graph (BERTopic + BERT Embeddings)")

    # Add legend with both clusters and queries
    plt.legend(
        handles=legend_elements,
        title="Topics & Queries",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=8
    )

    # Remove axes for cleaner look
    plt.axis('off')

    # Adjust layout to make room for legend
    plt.tight_layout()

    # Save the visualization
    plt.savefig(
        'knowledge_graph_visualization.png',
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()

def analyze_clusters(
    embeddings: np.ndarray,
    titles: List[str],
    min_cluster_size: int = 5
) -> Dict:
    """
    Analyze document clusters and return statistics.

    Args:
        embeddings: Document embeddings array
        titles: List of document titles
        min_cluster_size: Minimum size for HDBSCAN clusters

    Returns:
        Dictionary with cluster statistics
    """
    # Reduce dimensionality
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Cluster the documents
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_epsilon=0.1
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    # Analyze clusters
    clusters = defaultdict(list)
    for title, label in zip(titles, cluster_labels):
        clusters[label].append(title)

    # Calculate statistics
    stats = {
        'num_clusters': len([k for k in clusters.keys() if k != -1]),
        'noise_points': len(clusters[-1]) if -1 in clusters else 0,
        'cluster_sizes': {
            k: len(v) for k, v in clusters.items() if k != -1
        },
        'clusters': {
            k: v for k, v in clusters.items() if k != -1
        }
    }

    return stats
