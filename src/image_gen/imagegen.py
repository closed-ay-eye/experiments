from textwrap import dedent
from typing import List

import pandas as pd
from crewai import Agent, Task, Crew, Process
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
from pandas.core.series import Series
from langchain_openai import ChatOpenAI


class ImageGenResult(BaseModel):
    image_urls: List[str] = Field(default=[], description="The full URL for the image that illustrates the recipe step")


def _format_recipe(recipe: Series) -> str:
    name = recipe['name']
    ingredients = "\n".join((map(lambda x: f"- {x.replace("  ", "")}", eval(recipe['ingredients_raw_str']))))
    steps = "\n".join((map(lambda x: f"- {x}", eval(recipe['steps']))))
    return dedent(f"""Name: {name}\nIngredients:\n{ingredients}\n\nSteps:\n{steps}\n""")


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
            llm=ChatOpenAI(model="gpt-4o", temperature=0),
            backstory=(
                """ You are a really capable artist that can draw very photo-realistic illustration for each step of a recipe."""),
            tools=[self.__dalle],
            verbose=True,
            memory=True
        )

    def create_illustration(self, recipe: Series) -> ImageGenResult:
        recipe = _format_recipe(recipe)
        task = Task(
            description=f"Given the recipe below, create a photo-realistic illustration for each step of the recipe. "
                        f"Please keep the full URL for images as they were provided by the tool\n\n{recipe}",
            agent=self.__agent,
            expected_output="A list of URLs for the generated illustrations.Please keep the entire url for the "
                            "images, do not short them",
            output_pydantic=ImageGenResult
        )
        crew = Crew(
            agents=[self.__agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()


if __name__ == "__main__":
    df = pd.read_csv("../../dataset/recipes_w_search_terms.csv")
    a = Illustrator()
    print(a.create_illustration(df.iloc[0]))
