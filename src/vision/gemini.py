import google.generativeai as genai
import os
import PIL.Image


class Gemini:
    def __init__(self, google_api_key):
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel('gemini-pro-vision')

    def detect_ingredients(self, ingredients_picture):
        prompt = """This image contains ingredients for a food recipe. I would like you to identify all the ingredients
        in this image and return their names as a comma separated list."""
        return self.model.generate_content([prompt, ingredients_picture])


if __name__ == "__main__":
    api_key = os.getenv('GOOGLE_API_KEY')
    gemini = Gemini(api_key)
    image = PIL.Image.open('../../images/shrimp-tofu.webp')
    detection = gemini.detect_ingredients(image)
    print(f"Ingredients detected:{detection.text}")
