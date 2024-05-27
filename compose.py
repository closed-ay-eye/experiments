from pandas.core.series import Series


class RecipePromptComposer:
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
        Description: {recipe['description']}
        Number of servings: {recipe['servings']}
        Serving size: {recipe['serving_size']}
        Tags: {recipe['tags']}
        Search terms: {recipe['search_terms']}
        """
