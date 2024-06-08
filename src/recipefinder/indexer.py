import logging

import faiss
import numpy as np
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pandas import DataFrame
from pandas.core.series import Series

from src.recipefinder.embedding import EmbeddedCalculator

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class IngredientsIndexer:
    def __init__(self, dataframe: DataFrame, embedding_calculator: EmbeddedCalculator, faiss_index_path: str):
        self.__df = dataframe
        self.__ec = embedding_calculator
        self.__index_path = faiss_index_path

    def index(self):
        logging.info("Starting indexing")
        logging.info("Splitting ingredients list")
        logging.info(f"Indexing {len(self.__df.index)} recipes")
        self.__df['ingredients'] = self.__df['ingredients'].apply(lambda x: x.strip("[]").replace("'", "").split(', '))

        def embedding_with_progress(items: [str]):
            logging.info(f"Indexing recipe {embedding_with_progress.counter}")
            embedding_with_progress.counter += 1
            return self.__ec.for_list(items)

        embedding_with_progress.counter = 0

        self.__df['ingredient_embeddings'] = self.__df['ingredients'].apply(lambda x: embedding_with_progress(x))

        # Convert the embeddings to a suitable format for FAISS
        logging.info("Convert the embeddings to a suitable format for FAISS")
        embeddings = np.vstack(self.__df['ingredient_embeddings'].values)

        # Create and populate the FAISS index
        logging.info("Adding vectors to FAISS")
        d = embeddings.shape[1]  # Dimension of the embeddings
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        # Save the index
        logging.info("Saving the index")
        faiss.write_index(index, self.__index_path)


class IndexSearch:
    def __init__(self, embedding_calculator: EmbeddedCalculator, index_path: str, recipes_dataframe: DataFrame):
        self.__ec = embedding_calculator
        self.__index = faiss.read_index(index_path)
        self.__df = recipes_dataframe

    def search(self, ingredients: [str], n_results=5) -> Series:
        ingredients_embeddings = self.__ec.for_list(ingredients).reshape(1, -1)
        D, I = self.__index.search(ingredients_embeddings, n_results)
        return self.__df.iloc[I[0]]


def create_cached_embedder() -> Embeddings:
    embedding = OpenAIEmbeddings()
    store = LocalFileStore("./cache/")
    return CacheBackedEmbeddings.from_bytes_store(
        embedding, store, namespace=embedding.model
    )


def merge_faiss_indexes(first_path: str, second_path: str, result_path: str):
    index_1 = faiss.read_index(first_path)
    index_2 = faiss.read_index(second_path)
    index_1.merge_from(index_2)
    faiss.write_index(index_1, result_path)

