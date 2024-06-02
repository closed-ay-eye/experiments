import json
from pandas.core.series import Series
import re


def pretty_print_recipe(recipe: Series, rationale=""):
    print(f"Name: {recipe['name']}")
    if len(rationale) != 0: print(f"Rationale: {rationale}")
    print(f"Description: {recipe['description']}")
    ingredients = json.loads(recipe['ingredients_raw_str'])
    print("Ingredients:")
    for ingredient in ingredients:
        ing = re.sub("\s\s+", " ", ingredient)
        print(f"* {ing}")
