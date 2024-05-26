import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


class EmbeddedCalculator:
    """
     A class to compute the embeddings for a list of unordered strings
    """

    def __init__(self, embedder: Embeddings):
        self.embedder = embedder

    def __embedding_for_single_string(self, item: str) -> list[float]:
        print(f"Embedding for [{item}]")
        return self.embedder.embed_query(item)

    def __average_embeddings(self, embeddings):
        return np.mean(embeddings, axis=0)

    def for_list(self, items: [str]):
        embeddings = [self.__embedding_for_single_string(item) for item in items]
        return self.__average_embeddings(embeddings)


if __name__ == "__main__":
    embbeding = OpenAIEmbeddings()
    ec = EmbeddedCalculator(embbeding)
    a = ec.for_list(["tomate", "basil", "cheese"])
    b = ec.for_list(["basil", "cheese", "tomate"])
    c = ec.for_list(["basil", "cheese"])
    d = ec.for_list(["basil"])
    e = ec.for_list(["kotlin", "swift", "java"])
    print(compute_cosine_similarity(a, b))
    print(compute_cosine_similarity(a, c))
    print(compute_cosine_similarity(a, d))
    print(compute_cosine_similarity(a, e))
