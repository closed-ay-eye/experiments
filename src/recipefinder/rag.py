import logging
import json

from pandas.core.series import Series
from openai import OpenAI


class RecipePromptComposer:
    def __init__(self):
        self.system_prompt = "You are a helpful assistant that suggests recipes based on user preferences."

    def user_prompt_for_recipes(self, recipes: Series, user_input: str) -> str:
        recipe_list = []
        for i in range(len(recipes)):
            recipe_list.append(recipes.iloc[i])
        recipe_list_strings = "\n".join([self.__format_recipe(x) for x in recipe_list])
        prompt = f"""
        Based on user input, which of the following recipes would you recommend?
            
        {recipe_list_strings}
            
        User Input: {user_input}
            
        Please choose the best recipe based on the provided information and ensure that the output is in JSON format `{{recipe: <recipe's ID>}}`
        Add your rationale in the "rationale" key.
        """
        return prompt

    def __format_recipe(self, recipe: Series):
        return f"""
        Recipe ID: {recipe.name}
        Name: {recipe['name']}
        Description: {recipe['description']}
        Number of servings: {recipe['servings']}
        Serving size: {recipe['serving_size']}
        Tags: {recipe['tags']}
        Search terms: {recipe['search_terms']}
        """


class OpenAiQuery:

    def __init__(self, openai_client: OpenAI, composer: RecipePromptComposer, model: str = "gpt-4o",
                 temperature: float = 0.5):
        self.__model = model
        self.__composer = composer
        self.__temperature = temperature
        self.__client = openai_client

    def do_rag_query(self, recipes: Series, user_input: str):
        user_prompt = self.__composer.user_prompt_for_recipes(recipes, user_input)
        response = self.__client.chat.completions.create(
            model=self.__model,
            messages=[
                {"role": "system", "content": self.__composer.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        response_json = {}
        try:
            response_json = json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError as e:
            logging.error("Failed to decode OpenAI response")

        return response_json