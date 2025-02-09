from sentence_transformers import SentenceTransformer
from topic_model import WikipediaTopicModel
from bertopic import BERTopic
from config import config

def add_new_post(post: str, topic_model: BERTopic, posts_path=config.posts_path):
    """
    saves a new post, and determines its topic vector

    Args:
        text: string, the text of the post
        topic_model: BERTopic, the topic model to use for determining the post's topic representation
        topics: List of topics and their average BERT embedding.
    """

    embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-dot-v1', device=config.device)

    # Get the BERT embedding of the post
    # post_embedding = embedding_model.encode(text)


def update_post(post_id):
    """
    updates a post's impressions or like count
    """


if __name__ == "__main__":
    pass
#     add_new_post(
#         """2021–22 CF Fuenlabrada season
# The 2021–22 season was the 47th season in CF Fuenlabrada's history and the club's third consecutive season in the second division of Spanish football. In addition to competing in the domestic league, Fuenlabrada participated in this season's edition of the Copa del Rey.""")










