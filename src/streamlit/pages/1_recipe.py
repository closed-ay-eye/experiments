import streamlit as st
from PIL import Image
from src.streamlit.session import SessionKeys
from src.vision.gemini import Gemini


if SessionKeys.GEMINI_MODEL not in st.session_state:
    st.session_state[SessionKeys.GEMINI_MODEL] = Gemini()

col1, col2 = st.columns(2)


def display_image_column(image):
    with col1:
        st.image(
            image,
            caption="You just uploaded an image!",
        )

        tmp_img_path = f"temp_uploaded_image.{image.format}"
        image.save(tmp_img_path)

        return tmp_img_path


def display_result_column(tmp_img_path):
    with col2:
        gemini = st.session_state.gemini_model
        detection = gemini.detect_ingredients(tmp_img_path)
        st.markdown(detection.content)


def display(image):
    image_path = display_image_column(image)
    display_result_column(image_path)


img_file_buffer = st.camera_input(label="Take an ingredients picture")
img_upload_buffer = st.file_uploader(label="Or upload an image")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    display(img)

if img_upload_buffer is not None:
    img = Image.open(img_upload_buffer)
    display(img)


