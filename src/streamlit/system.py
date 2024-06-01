from dataclasses import dataclass

import pandas as pd
from PIL import Image
from rx.subject import Subject
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.recipefinder.embedding import EmbeddedCalculator
from src.recipefinder.indexer import create_cached_embedder, IndexSearch
from src.recipefinder.rag import RecipePromptComposer, LangChainQuery
from src.vision.gemini import Gemini


@dataclass
class DisplayState:
    image: Image
    recipe_text: str = ""
    loading_message: str = ""


class SystemModel:

    def __init__(
            self,
            dataset_path="dataset/recipes_w_search_terms.csv",
            index_path="indexes/ingredient_index_14400",
    ):
        self.df = pd.read_csv(dataset_path)
        self.cached_embedder = create_cached_embedder()
        ec = EmbeddedCalculator(self.cached_embedder)
        self.index = IndexSearch(ec, index_path, self.df)
        self.composer = RecipePromptComposer()
        self.query = LangChainQuery(self.composer)
        self.gemini = Gemini()
        self.subject = Subject()

    def observe_events(self):
        return self.subject

    def detect_ingredients(self, tmp_img_path):
        return self.gemini.detect_ingredients(tmp_img_path)

    def search_by_ingredient(self, ingredients: [str], user_prompt="I don't have a preference for the recipe, pick any."):
        print(ingredients)
        search_results = self.index.search(ingredients)
        response = self.query.do_rag_query(search_results, user_prompt)
        if response.recipe is not None:
            recipe = self.df.iloc[response.recipe]
            return f"{recipe}\n {response.rationale}"
        else:
            return f"Sorry, we could not find a recipe. {response.rationale}"

    def on_user_input(self, camera_image: UploadedFile, uploaded_image: UploadedFile, user_prompt: str):
        image = None

        if camera_image is not None:
            image = Image.open(camera_image)

        if uploaded_image is not None:
            image = Image.open(uploaded_image)

        if image is not None:
            tmp_img_path = f"temp_uploaded_image.{image.format}"
            image.save(tmp_img_path)

            self.subject.on_next(
                DisplayState(
                    image=image,
                    loading_message="LOAding..."
                )
            )

            content = self.detect_ingredients(tmp_img_path)
            response = self.search_by_ingredient(content.ingredients)

            self.subject.on_next(
                DisplayState(
                    image=image,
                    recipe_text=response
                )
            )
