from dataclasses import dataclass, field
from typing import Optional,List

from PIL import Image


@dataclass
class ProcessingState:
    uploaded_image: Image
    loading_message: str = ""


@dataclass
class IllustratedStep:
    recipe_step: str
    recipe_image_url: str


@dataclass
class DisplayState:
    uploaded_image: Image
    recipe_name: str
    recipe_ingredients: str
    recipe_steps: list[IllustratedStep]
    recipe_text: str = ""
    recipe_image_url: str = ""
    audio_ingredients: Optional[bytes] = None
    steps_audio: List[Optional[bytes]] = field(default_factory=list)


@dataclass
class WaitingInputState:
    pass
