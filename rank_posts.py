import numpy as np
import scipy as sp


def rank(posts, preferences, n_posts, temperature=1.0):
    """
    Ranks posts according to a user's preferences.
    
    Args:
        posts: array of shape (n, d), where d is the dimension of topics
        preferences: array of shape (d,) of user preferences
        n_posts: number of posts to return
    """

    # Dot product between posts and preferences, then softmax

    out = np.dot(posts, preferences)

    normalized = sp.special.softmax(out / temperature)

    # randomly sample n_posts from the normalized distribution, without replacement
    indices = np.random.choice(len(posts), size=n_posts, replace=False, p=normalized)

    return indices



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

    print(rank(post_topics, preferences, n_posts, temperature))

