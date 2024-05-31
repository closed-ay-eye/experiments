#!/usr/bin/env python3

import pandas as pd

from embedding import EmbeddedCalculator
from indexer import create_cached_embedder, IndexSearch
from rag import RecipePromptComposer, LangChainQuery
from utils import pretty_print_recipe

if __name__ == "__main__":
    # Loading
    df = pd.read_csv("dataset/recipes_w_search_terms.csv")
    cached_embedder = create_cached_embedder()
    ec = EmbeddedCalculator(cached_embedder)
    index = IndexSearch(ec, "indexes/ingredient_index_7200", df)

    print("Hi ! I am test python script to test the recipe suggestion engine.")
    ingredients = []
    print("Please enter ingredients one by one when asked. Just press enter to finish entering the ingredients")
    ingredient = input("> ").strip()
    while len(ingredient) != 0:
        ingredients.append(ingredient)
        ingredient = input("> ").strip()
    if len(ingredients) == 0:
        print("Sorry, you need to enter ingredients")
        exit(1)
    print("Nice! Now that we have the ingredients tell me what you want for your recipe: ")
    user_prompt = input("> ")
    if len(user_prompt) == 0:
        user_prompt = "I don't have a preference for the recipe, pick any."
    search_results = index.search(ingredients)
    composer = RecipePromptComposer()
    query = LangChainQuery(composer)
    response = query.do_rag_query(search_results, user_prompt)
    if response.recipe is not None:
        recipe = df.iloc[response.recipe]
        pretty_print_recipe(recipe, response.rationale)
    else:
        print(f"Sorry, we could not find a recipe. {response.rationale}")
    # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # query = OpenAiQuery(client, composer)
    # response = query.do_rag_query(search_results, user_prompt)
    # recipe = df.iloc[response['recipe']]
    # print("Here's your recipe: \n\n")
    # pretty_print_recipe(recipe, response['rationale'])