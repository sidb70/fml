import tensorflow_datasets as tfds
import spacy

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from tqdm import tqdm

from config import config


class WikipediaTopicModel:

    def __init__(self, model_path: str=config.model_path, docs_path: str=config.docs_path, device: str = config.device):
        self.model_path = model_path
        self.docs_path = docs_path
        self.device = device


    def train(self, sample_size=5000) -> BERTopic:
        """
        Trains the topic model
        """
        texts = self.load_and_process_wikipedia_docs(sample_size=sample_size)
        return self.train_topic_model(texts)
    

    def load(self) -> BERTopic:
        """
        Loads the pre-trained topic model
        """
        try:
            return BERTopic.load(self.model_path)
        except FileNotFoundError:
            print("No model found, training a new model")
            return self.train()


    def load_and_process_wikipedia_docs(self, sample_size=config.sample_size):
        """
        Processes and cleans the text data
        """

        # check if processed texts already exist
        try:
            return self.load_processed_texts()
        except FileNotFoundError:
            pass

        # 1. Load Wikipedia Data
        print("Downloading Wikipedia Data")
        ds = tfds.load('wikipedia/20200301.en', split='train', shuffle_files=True)
        docs = []
        for example in ds.take(sample_size): # Reduce for quick experimentation
            docs.append(example['text'].numpy().decode('utf-8'))

        # 2. Preprocess Data (simple version)
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ValueError("Run 'python -m spacy download en_core_web_sm'")
        
        nlp = spacy.load("en_core_web_sm")

        processed_docs = []
        for text in tqdm(docs, desc="Preprocessing Documents"):
            doc = nlp(text)
            processed_text = " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
            processed_docs.append(processed_text)

        # save list of documents to file
        with open(self.docs_path, "w") as f:
            for text in processed_docs:
                f.write(text + "\n")

        return processed_docs
    

    def train_topic_model(self, docs=None) -> BERTopic:

        if docs is None:
            try:
                docs = self.load_processed_texts()
            except FileNotFoundError:
                raise ValueError("No processed texts found, please provide texts to train the model")

        # Create Document Embeddings using BERT
        embedder = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-dot-v1', device=self.device)

        # Reduce dimensionality
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

        # Cluster reduced embeddings
        hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        # Tokenize topics
        vectorizer_model = CountVectorizer(stop_words="english")

        # Create topic representation
        ctfidf_model = ClassTfidfTransformer()

        # Create topic representation
        representation_model = KeyBERTInspired()

        # Define BERTopic model
        topic_model = BERTopic(
            embedding_model=embedder, 
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            min_topic_size=15
        )

        # Fit BERTopic model
        print("Fitting BERTopic model")
        topics, probs = topic_model.fit_transform(docs)

        # Save model
        topic_model.save(self.model_path)

        return topic_model


    def load_processed_texts(self) -> list[str]:
        """
        Loads the processed text from the file
        """
        with open(self.docs_path, "r") as f:
            return f.readlines()
    

if __name__ == "__main__":
    topic_model = WikipediaTopicModel('data/topic_model', device='mps')
    topic_model.train(5000)

