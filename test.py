import tensorflow_datasets as tfds
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Wikipedia Data
def load_wikipedia_data(sample_size=100000):
    ds = tfds.load('wikipedia/20200301.en', split='train', shuffle_files=True)
    texts = []
    for example in ds.take(sample_size):  # Reduce for quick experimentation
        texts.append(example['text'].numpy().decode('utf-8'))
    return texts

# 2. Preprocess Data (simple version)
texts = load_wikipedia_data()

# 3. Create Document Embeddings using BERT
embedder = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-dot-v1', device='cuda')
embeddings = embedder.encode(texts, show_progress_bar=True)

# 4. Cluster Topics with BERTopic
topic_model = BERTopic(embedding_model=embedder, min_topic_size=15)
topics, _ = topic_model.fit_transform(texts, embeddings)

# 5. Create Knowledge Graph Nodes (Topics) - FIXED
topic_info = topic_model.get_topic_info()
filtered_topics = topic_info[topic_info['Topic'] != -1].copy()  # Filter out outliers
topic_embeddings = np.array([topic_model.topic_embeddings_[topic] 
                           for topic in filtered_topics['Topic']])

# 6. Calculate Cosine Similarity for Edges (unchanged)
similarity_matrix = cosine_similarity(topic_embeddings)

# 7. Build Graph - FIXED
G = nx.Graph()
threshold = 0.65

# Add nodes with metadata using FILTERED topics
for idx, row in filtered_topics.iterrows():
    G.add_node(row['Topic'],
               label=f"Topic {row['Topic']}",
               keywords=", ".join([word[0] for word in topic_model.get_topic(row['Topic'])]),
               size=row['Count'])  # Verify column name is 'Count'

# Add edges using FILTERED indices
for i in range(len(filtered_topics)):
    for j in range(i+1, len(filtered_topics)):
        if similarity_matrix[i][j] > threshold:
            G.add_edge(filtered_topics.iloc[i]['Topic'], 
                      filtered_topics.iloc[j]['Topic'],
                      weight=similarity_matrix[i][j])

# save graph
nx.write_graphml(G, 'my_graph.graphml')


# 8. Visualize Knowledge Graph
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
plt.savefig("wikipedia_knowledge_graph.png")
plt.show()