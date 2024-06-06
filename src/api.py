from pydantic import BaseModel, Field
from typing import List, Optional

from fastapi import FastAPI
import pandas as pd

from copywriter.copywriter import Copywriter, RecipeClassScript
from photo_scrapper import retrieve_recipe_photo
from recipefinder.embedding import EmbeddedCalculator
from recipefinder.indexer import create_cached_embedder, IndexSearch
from recipefinder.rag import LangChainQuery


class IngredientRequest(BaseModel):
    ingredients: List[str] = Field(description="List of ingredients")
    prompt: Optional[str] = Field(description="The prompt for the recipe")


class RecipeResult(BaseModel):
    name: str = Field(description="The name of the recipe")
    image: Optional[str] = Field(default=None, description="The URL for recipe's image")
    description: str = Field(description="The name of the recipe")
    ingredients: List[str] = Field(description="The list of ingredients")
    steps: List[str] = Field(description="The list recipe steps")
    servings: str = Field(default="", description="The number of servings for this recipe")
    serving_size: str = Field(default="", description="The serving size")
    script: RecipeClassScript = Field(description="The script for narration and image generation")


class ErrorResult(BaseModel):
    code: int = Field(default=0, description="The error code")
    description: str = Field(default="", description="The error description")


class RecipeRequestResult(BaseModel):
    recipe: Optional[RecipeResult] = Field(default=None, description="The recipe")
    error: Optional[ErrorResult] = Field(default=None, description="The error for this request")


app = FastAPI()
df = pd.read_csv("dataset/recipes_w_search_terms.csv")
cached_embedder = create_cached_embedder()
ec = EmbeddedCalculator(cached_embedder)
index = IndexSearch(ec, "indexes/ingredient_index_14400", df)


@app.post("/recipes/find")
def find_recipe(request: IngredientRequest):
    if len(request.ingredients) > 0:
        search_results = index.search(request.ingredients)
        user_prompt = request.prompt if len(request.prompt.strip()) > 0 else "Pick any"
        chosen_recipe = LangChainQuery().do_rag_query(search_results, user_prompt)
        if chosen_recipe.recipe is not None:
            recipe_dataframe = df.iloc[chosen_recipe.recipe]
            script = Copywriter().create_script(recipe_dataframe)
            image = retrieve_recipe_photo(recipe_dataframe['id'])
            return RecipeRequestResult(recipe=RecipeResult(
                name=recipe_dataframe['name'],
                image=image,
                description=recipe_dataframe['description'],
                ingredients=eval(recipe_dataframe['ingredients_raw_str']),
                # steps=eval(recipe_dataframe['steps']),
                # servings=recipe_dataframe['servings'],
                serving_size=recipe_dataframe['serving_size'],
                script=script
            ))
        else:
            pass
    else:
        pass
