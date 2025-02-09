import numpy as np
import scipy as sp
from typing import List
from bertopic import BERTopic


def rank_posts(all_posts: List[str], preferences: np.ndarray, topic_model: BERTopic, top_n_posts, temperature=1.0):
    """
    Ranks the top n posts according to a user's preferences.
    
    Args:
        posts: list of strings, the text of the posts
        preferences: array of shape (d,) of user preferences
        topic_model: BERTopic, the topic model to use for determining the post's topic representation
        n_posts: number of posts to return
    """


    post_representations = np.array([generate_post_representation(post, topic_model) for post in all_posts]) # array of shape (n_posts, n_topics)

    # Dot product between posts and preferences, then softmax
    out = np.dot(post_representations, preferences)
    normalized = sp.special.softmax(out / temperature)

    # randomly sample n_posts from the normalized distribution, without replacement
    indices = np.random.choice(len(all_posts), size=top_n_posts, replace=False, p=normalized)

    return indices


def generate_post_representation(post, topic_model: BERTopic):
    """
    Returns a vector of length n_topics, where each element represents the post's relevance to that topic.

    Args:
        post: string, the text of the post
        topic_model: BERTopic, the topic model to use for determining the post's topic representation

    Returns:
        post_representation: array of shape (n_topics,) representing the post's relevance
    """

    return topic_model.transform([post])



if __name__ == "__main__":

    post_topics = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.3],
        [0.3, 0.1, 0.2],
        [0.1, 0.3, 0.2],
        [0.2, 0.3, 0.1]
    ])

    preferences = np.array([0.1, 0.2, 0.7])

    n_posts = 3

    temperature = 1.0

    print(rank_posts(post_topics, preferences, n_posts, temperature))
