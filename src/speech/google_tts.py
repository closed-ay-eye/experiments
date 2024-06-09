import logging
import os

from google.cloud import texttospeech


class GoogleTTS:
    """
        Text-to-speech using Google Cloud
        Needs GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to service account key json file
    """
    def __init__(self):
        self.create_tts_secret_file()
        self.__client = texttospeech.TextToSpeechClient()
        self.__voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Journey-F")
        self.__audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                                                       speaking_rate=1)

    def create_tts_secret_file(self):
        tts_secret_content = os.environ['GOOGLE_APPLICATION_CREDENTIALS_CONTENT']
        text_file = open("GOOGLE_APPLICATION_CREDENTIALS.json", "w")
        text_file.write(tts_secret_content)
        text_file.close()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GOOGLE_APPLICATION_CREDENTIALS.json'

    def for_text(self, text: str)-> bytes:
        logging.info(f"Creating TTS for {text}")
        input_text = texttospeech.SynthesisInput(text=text)
        response = self.__client.synthesize_speech(
            request={"input": input_text, "voice": self.__voice, "audio_config": self.__audio_config}
        )
        return response.audio_content

if __name__ == "__main__":
    tts = GoogleTTS()
    audio_bytes = tts.for_text("Pity the world, or else this glutton be—To eat the world’s due, by the grave and thee")

    # The response's audio_content is binary.
    with open("../output.mp3", "wb") as out:
        out.write(audio_bytes)
        print('Audio content written to file "output.mp3"')