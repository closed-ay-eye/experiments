from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class Gemini:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model='gemini-pro-vision')

    def detect_ingredients(self, ingredients_picture_path):
        prompt = """This image contains ingredients for a food recipe. I would like you to identify all the ingredients
        in this image and return their names in a MARKDOWN format. If there are no ingredients in the picture, I would
        like you to return an empty MARKDOWN."""

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": ingredients_picture_path,
                },
            ]
        )

        return self.model.invoke([message])


if __name__ == "__main__":
    gemini = Gemini()
    image_path = 'images/shrimp-tofu.webp'

    detection = gemini.detect_ingredients(image_path)
    print(f"Ingredients detected:{detection.content}")
