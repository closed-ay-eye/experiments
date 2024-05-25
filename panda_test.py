import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("dataset/recipes_w_search_terms.csv")
    print(df['ingredients'][0].strip("[]").replace("'", "").split(', ')[2])