import pandas as pd
from pandas.core.series import Series
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List


class RecipeClassScript(BaseModel):
    ingredients: str = Field(description="The whole ingredients list as said by culinary teacher")
    steps: List[str] = Field(description="Each os the step as said by the culinary teacher")


class Copywriter:
    def __init__(self,
                 llm: BaseChatModel = ChatOpenAI(model="gpt-4o", temperature=0)):
        self.__chat = llm
        self.__prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that suggests recipes based on user preferences."),
            ("human", "{user_prompt}")
        ])
        self.__parser = PydanticOutputParser(pydantic_object=RecipeClassScript)
        self.__chain = (self.__prompt | self.__chat | self.__parser).with_config({"run_name": "Copywriter"})

    def _format_recipe(self, recipe: Series) -> str:
        name = recipe['name']
        ingredients = "\n".join((map(lambda x: f"- {x.replace("  ", "")}", eval(recipe['ingredients_raw_str']))))
        steps = "\n".join((map(lambda x: f"- {x}", eval(recipe['steps']))))
        return f"""Name: {name}\nIngredients:\n{ingredients}\n\nSteps:\n{steps}\n"""

    def create_script(self, recipe: Series) -> RecipeClassScript:
        formatted_recipe = self._format_recipe(recipe)
        user_prompt = (f"You are a famous and very didactic culinary teacher. Rewrite the ingredients list for the "
                       f"recipe below as if you teaching the recipe to your dear apprentice. Do the same thing for "
                       f"each step of the recipe. Here is the recipe you will teach:\n\n{formatted_recipe}\n\n"
                       f"{self.__parser.get_format_instructions()}")
        return self.__chain.invoke({"user_prompt": user_prompt})


if __name__ == "__main__":
    df = pd.read_csv("../../dataset/recipes_w_search_terms.csv")
    cw = Copywriter()
    print(cw.create_script(df.iloc[0]))