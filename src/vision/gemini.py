from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List
from langchain.pydantic_v1 import BaseModel, Field
import base64
import mimetypes
import os


class IngredientList(BaseModel):
    ingredients: List[str] = Field(default=[], description="A list of ingredients")


def image_to_base64_url(image_path):
    # Detect the MIME type of the image
    mime_type, _ = mimetypes.guess_type(image_path)
    # Open the image file
    with open(image_path, "rb") as image_file:
        # Read the image file and encode it to base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # Form the base64 URL
        base64_url = f"data:{mime_type};base64,{encoded_string}"
        return base64_url


class Gemini:
    def __init__(self):
        self.__model = ChatGoogleGenerativeAI(model='gemini-pro-vision')
        self.__parser = PydanticOutputParser(pydantic_object=IngredientList)

    def detect_ingredients(self, ingredients_picture_path):
        image_base64_url_encoded = image_to_base64_url(ingredients_picture_path)
        prompt = f"""This image contains ingredients for a food recipe. I would like you to identify all the ingredients
        in this image and return their names in as list. If no ingredient is found in the image return an empty 
        list. Ingredients should be in lowercase. Do not mention brand names. {self.__parser.get_format_instructions()}"""

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": image_base64_url_encoded,
                },
            ]
        )

        return (self.__model | self.__parser).with_config({"run_name": "Gemini Vision Ingredient"}).invoke([message])


if __name__ == "__main__":
    gemini = Gemini()
    image_folder = "../../images"
    for file in os.listdir(image_folder):
        detection = gemini.detect_ingredients(f"{image_folder}/{file}")
        print(f"{file} -> {detection}")
