import streamlit as st
from rx.disposable import Disposable

from src.streamlit.session import SessionKeys
from src.streamlit.system import SystemModel
from src.streamlit.state import DisplayState, WaitingInputState, ProcessingState

if SessionKeys.SYSTEM_MODEL not in st.session_state:
    st.session_state[SessionKeys.SYSTEM_MODEL] = SystemModel()

model: SystemModel = st.session_state[SessionKeys.SYSTEM_MODEL]

input_holder = st.empty()
answer_holder = st.empty()


def handle_state(state):
    answer_holder.empty()
    input_holder.empty()

    match state:
        case ProcessingState():
            display_processing(state)

        case DisplayState():
            display_answer(state)

        case WaitingInputState():
            display_waiting_input()


def display_processing(state: ProcessingState):
    with answer_holder.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                state.uploaded_image,
                caption="You just uploaded an image!",
            )

        with col2:
            st.markdown(state.loading_message)


def display_answer(state: DisplayState):
    with answer_holder.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                state.uploaded_image,
                caption="You just uploaded an image!",
            )

        with col2:
            st.markdown(state.recipe_text)
            if state.recipe_image_url:
                st.image(state.recipe_image_url)
            if st.button('Restart'):
                model.on_return_to_start()
                st.rerun()


def display_waiting_input():
    with input_holder.container():
        img_file_buffer = st.camera_input(label="Take an ingredients picture")
        img_upload_buffer = st.file_uploader(label="Or upload an image")
        user_prompt = st.text_area(label="Based on the ingredients from the picture tell me what you want for your recipe")

        model.on_image_inserted(img_file_buffer, img_upload_buffer, user_prompt)


def observe_model():
    return model.observe_events().subscribe(
        on_next=lambda event: handle_state(event),
        on_error=lambda e: print(f"Error: {e}"),
        on_completed=lambda: print("Stream completed")
    )


if SessionKeys.DISPOSABLE in st.session_state:
    disposable: Disposable = st.session_state[SessionKeys.DISPOSABLE]
    disposable.dispose()

st.session_state[SessionKeys.DISPOSABLE] = observe_model()
