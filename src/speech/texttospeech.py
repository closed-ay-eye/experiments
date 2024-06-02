from langchain_community.tools import ElevenLabsText2SpeechTool
from crewai import Agent, Task, Crew, Process
import os
from langchain_core.tools import Tool
import pandas as pd
from pandas.core.series import Series

# Replace your ELEVEN_API_KEY here
# os.environ["ELEVEN_API_KEY"] = ""

def _format_recipe(recipe: Series) -> str:
    name = recipe['name']
    ingredients = "\n".join((map(lambda x: f"- {x.replace('  ', '')}", eval(recipe['ingredients_raw_str']))))
    steps = "\n".join((map(lambda x: f"- {x}", eval(recipe['steps']))))
    return steps
    # return dedent(f"""Name: {name}\nIngredients:\n{ingredients}\n\nSteps:\n{steps}\n""")

class TextToSpeech:
    def __init__(self):
        tts = ElevenLabsText2SpeechTool()

        # can call the following to stream instead: tts.stream_speech(text_to_speak)
        self.__text2speech_tool = Tool(
            name="Intermediate Answer",
            func=tts.run,
            description="Input is text",
        )

        self.__agent = Agent(
            role='Speaker',
            goal='You take {text} and convert it to speech',
            backstory='You take {text} and convert it to speech',
            tools=[self.__text2speech_tool],
            verbose=True
        )

    def createSpeech(self, recipe: Series):
        recipe = _format_recipe(recipe)
        task = Task(
            description="Given some {text}, convert it to speech",
            agent=self.__agent,
            expected_output="The speech",
        )
        crew = Crew(
            agents=[self.__agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff(inputs={"text": recipe})

if __name__ == "__main__":
    df = pd.read_csv("../../dataset/recipes_w_search_terms.csv")
    t2S = TextToSpeech()
    # t2S.createSpeech("Hello World, this is testing text to speech")
    t2S.createSpeech(df.iloc[0])
