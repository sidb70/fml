import tensorflow as tf
import tensorflow_datasets as tfds
import bertopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
import torch


# 1. Load Wikipedia Data
def load_wikipedia_data(sample_size=5000):
    ds = tfds.load('wikipedia/20200301.en', split='train', shuffle_files=True)
    texts = []
    for example in ds.take(sample_size):  # Reduce for quick experimentation
        texts.append(example['text'].numpy().decode('utf-8'))
    return texts

if __name__ == "__main__":

    # 2. Preprocess Data (simple version)
    texts = load_wikipedia_data()

    # 3. Create Document Embeddings using BERT
    embedder = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-dot-v1', device='mps')
    embeddings = embedder.encode(texts, show_progress_bar=True)

    # 4. Cluster Topics with BERTopic
    topic_model = BERTopic(embedding_model=embedder, min_topic_size=15)
    topics, _ = topic_model.fit_transform(texts, embeddings)

    # export the model
    topic_model.save("model")

