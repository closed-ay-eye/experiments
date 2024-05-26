import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pandas import DataFrame
from embedding import EmbeddedCalculator
import numpy as np
import faiss
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings


class IngredientsIndexer:
    def __init__(self, dataframe: DataFrame, embedding_calculator: EmbeddedCalculator, faiss_index_path: str):
        self.df = dataframe
        self.ec = embedding_calculator
        self.index_path = faiss_index_path

    def index(self):
        print("Starting indexing")
        print("Splitting ingredients list")
        self.df['ingredients'] = self.df['ingredients'].apply(lambda x: x.strip("[]").replace("'", "").split(', '))
        self.df['ingredient_embeddings'] = self.df['ingredients'].apply(lambda x: self.ec.for_list(x))

        # Convert the embeddings to a suitable format for FAISS
        print("Convert the embeddings to a suitable format for FAISS")
        embeddings = np.vstack(self.df['ingredient_embeddings'].values)

        # Create and populate the FAISS index
        print("Adding vectors to FAISS")
        d = embeddings.shape[1]  # Dimension of the embeddings
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        # Save the index
        print("Saving the index")
        faiss.write_index(index, self.index_path)


if __name__ == "__main__":
    print("Reading panda")
    df = pd.read_csv("dataset/recipes_w_search_terms.csv")
    df = df.iloc[:30]
    print("Ready to index")
    store = LocalFileStore("./cache/")
    embedding = OpenAIEmbeddings()
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embedding, store, namespace=embedding.model
    )
    ec = EmbeddedCalculator(cached_embedder)
    indexer = IngredientsIndexer(df, ec, "my_index")
    indexer.index()
    faiss_index = faiss.read_index("my_index")
    emd = ec.for_list(['chicken breasts']).reshape(1, -1)
    D, I = faiss_index.search(emd, k=5)
    print(df.iloc[I[0]]['description'][6])
