from dataclasses import dataclass

from PIL import Image


@dataclass
class ProcessingState:
    uploaded_image: Image
    loading_message: str = ""


@dataclass
class DisplayState:
    uploaded_image: Image
    recipe_text: str = ""
    recipe_image_url: str = ""


@dataclass
class WaitingInputState:
    pass
