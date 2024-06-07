import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from dataclasses import dataclass

import pandas as pd
from PIL import Image
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from pandas.core.series import Series
from rx.subject import BehaviorSubject
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.copywriter.copywriter import Copywriter, RecipeClassScript
from src.photo_scrapper import retrieve_recipe_photo
from src.recipefinder.embedding import EmbeddedCalculator
from src.recipefinder.indexer import create_cached_embedder, IndexSearch
from src.recipefinder.rag import RecipePromptComposer, LangChainQuery
from src.streamlit.state import DisplayState, WaitingInputState, ProcessingState, IllustratedStep
from src.vision.gemini import Gemini
from src.speech.google_tts import GoogleTTS
from concurrent.futures import ThreadPoolExecutor, as_completed

tts = GoogleTTS()


@dataclass
class RecipeResponse:
    answer: str = ""
    recipe_dataframe: Series = None
    recipe_id: int = -1
    recipe_name: str = ""
    recipe_steps: list[str] = ()


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
        self.copywriter = Copywriter()
        self.subject = BehaviorSubject(WaitingInputState())

    def observe_events(self):
        return self.subject

    def detect_ingredients(self, tmp_img_path):
        return self.gemini.detect_ingredients(tmp_img_path)

    def search_by_ingredient(self, ingredients: [str],
                             user_prompt="I don't have a preference for the recipe, pick any."):
        print(ingredients)
        search_results = self.index.search(ingredients)
        response = self.query.do_rag_query(search_results, user_prompt)
        if response.recipe is not None:
            recipe = self.df.iloc[response.recipe]
            return RecipeResponse(
                answer=f"{recipe}\n {response.rationale}",
                recipe_dataframe=recipe,
                recipe_id=recipe['id'],
                recipe_name=recipe['name'],
                recipe_steps=eval(recipe['steps']),
            )
        else:
            return RecipeResponse(
                answer=f"Sorry, we could not find a recipe. {response.rationale}",
            )

    def generate_script(self, recipe_dataframe):
        return self.copywriter.create_script(recipe_dataframe)

    def generate_script_images(self, script: RecipeClassScript):
        images_url = map_strings_to_bytes(lambda x: DallEAPIWrapper(model="dall-e-3").run(x), script.steps_illustration)
        return images_url

    def build_illustrated_steps(self, recipe_script: RecipeClassScript, images_url: list[str]):
        i = 0
        illustrated_steps: list[IllustratedStep] = []
        for step in recipe_script.steps:
            image_url = None
            if i < len(images_url):
                image_url = images_url[i]

            illustrated_steps.append(
                IllustratedStep(
                    recipe_step=f"- {step}\n",
                    recipe_image_url=image_url,
                )
            )

            i += 1
        return illustrated_steps

    def on_image_inserted(self, camera_image: UploadedFile, uploaded_image: UploadedFile, user_prompt: str):
        image = None

        if camera_image is not None:
            image = Image.open(camera_image)

        if uploaded_image is not None:
            image = Image.open(uploaded_image)

        if image is not None:
            tmp_img_path = f"temp_uploaded_image.{image.format}"
            image.save(tmp_img_path)

            self.subject.on_next(
                ProcessingState(
                    uploaded_image=image,
                    loading_message="Loading..."
                )
            )

            content = self.detect_ingredients(tmp_img_path)
            if len(content.ingredients) == 0:
                self.subject.on_next(
                    DisplayState(
                        uploaded_image=image,
                        recipe_name="",
                        recipe_steps=[],
                        recipe_text="No Ingredients Found!",
                    )
                )
            else:
                response = self.search_by_ingredient(ingredients=content.ingredients, user_prompt=user_prompt)
                recipe_script = self.generate_script(response.recipe_dataframe)
                images = self.generate_script_images(recipe_script)
                audio_ingredients = tts.for_text(recipe_script.ingredients)
                illustrated_steps = self.build_illustrated_steps(recipe_script, images)

                self.subject.on_next(
                    DisplayState(
                        uploaded_image=image,
                        recipe_name=response.recipe_name,
                        recipe_ingredients=eval(response.recipe_dataframe['ingredients_raw_str']),
                        recipe_steps=illustrated_steps,
                        recipe_text=response.answer,
                        recipe_image_url=retrieve_recipe_photo(str(response.recipe_id)),
                        audio_ingredients=audio_ingredients,
                        steps_audio=map_strings_to_bytes(lambda x: tts.for_text(x), recipe_script.steps)
                        # steps_audio=[tts.for_text(x) for x in recipe_script.steps]
                    )
                )

    def on_return_to_start(self):
        self.subject.on_next(
            WaitingInputState()
        )


def map_strings_to_bytes(f, strings):
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(f,s) for s in strings]

        # Collect results as they complete
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"An error occurred: {exc}")

        return results