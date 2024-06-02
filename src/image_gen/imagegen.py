from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser
from pandas.core.series import Series
import pandas as pd
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from crewai import Agent, Task, Crew, Process

from langchain_core.tools import Tool


class ImageGenResult(BaseModel):
    image_urls: List[str] = Field(default=[], description="The full URL for the image that illustrates the recipe step")


class Illustrator:
    def __init__(self):
        self.__dalle = Tool(
            "Dall-E-Image-Generator",
            DallEAPIWrapper(model="dall-e-3").run,
            "A wrapper around OpenAI DALL-E API. Useful for when you need to generate images from a text description. Input "
            "should be an image description.",
        )

        self.__agent = Agent(
            role="Illustrator",
            goal="Create a photo-realistic illustration for each the recipe's steps",
            backstory=(
                """ You are a really capable artist that can draw very photo-realistic illustration for each step of a recipe."""),
            tools=[self.__dalle],
            verbose=True,
            memory=True
        )


    def format_recipe(self, recipe: Series) -> str:
        name = recipe['name']
        ingredients = "\n".join((map(lambda x: f"- {x}", eval(recipe['ingredients_raw_str']))))
        steps = "\n".join((map(lambda x: f"- {x}",eval(recipe['steps']))))
        return f"""Name: {name}
        Ingredients:
        {ingredients}
        
        Step:
        {steps}
           
        """





if __name__ == "__main__":
    df = pd.read_csv("../../dataset/recipes_w_search_terms.csv")
    a = Illustrator()
    print(a.format_recipe(df.iloc[0]))
