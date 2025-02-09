import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



def create_graph(topic_model, threshold=0.75, save_path=None):

    # 1. Get Topic Embeddings
    topic_info = topic_model.get_topic_info()
    filtered_topics = topic_info[topic_info['Topic'] != -1].copy()  # Filter out outliers
    topic_embeddings = np.array([topic_model.topic_embeddings_[topic] 
                            for topic in filtered_topics['Topic']])

    # 2. Calculate Cosine Similarity for Edges (unchanged)
    similarity_matrix = cosine_similarity(topic_embeddings)


    # 3. Build Graph 
    G = nx.Graph()

    # Add nodes with metadata using FILTERED topics
    for idx, row in filtered_topics.iterrows():
        G.add_node(row['Topic'],
                label=f"Topic {row['Topic']}",
                keywords=", ".join([word[0] for word in topic_model.get_topic(row['Topic'])]),
                size=row['Count'])  # Verify column name is 'Count'

    # 4. Add edges using FILTERED indices
    for i in range(len(filtered_topics)):
        for j in range(i+1, len(filtered_topics)):
            if similarity_matrix[i][j] > threshold:
                G.add_edge(filtered_topics.iloc[i]['Topic'], 
                        filtered_topics.iloc[j]['Topic'],
                        weight=similarity_matrix[i][j])

    if save_path:
        nx.write_graphml(G, save_path)
    return G

def visualize_graph(G, save_path=None):
    #  Visualize Knowledge Graph
    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G, k=0.5)

    node_sizes = [G.nodes[node]['size']*10 for node in G.nodes]
    edge_weights = [G.edges[edge]['weight']*2 for edge in G.edges]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.2)
    nx.draw_networkx_labels(G, pos, 
                            labels={node:G.nodes[node]['label'] for node in G.nodes},
                            font_size=8)

    # Add keyword annotations
    for node in G.nodes:
        plt.annotate(G.nodes[node]['keywords'], 
                    xy=pos[node], 
                    xytext=(10, -10),
                    textcoords='offset points',
                    fontsize=6,
                    alpha=0.7)

    plt.title("Wikipedia Knowledge Graph (BERTopic + BERT Embeddings)")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()