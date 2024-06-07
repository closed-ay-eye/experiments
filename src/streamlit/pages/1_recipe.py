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
        st.image(
            state.uploaded_image,
            caption="You just uploaded an image!",
        )
        st.markdown(state.loading_message)


def display_answer(state: DisplayState):
    with answer_holder.container():
        st.title(state.recipe_name)

        if state.recipe_image_url:
            st.image(state.recipe_image_url)
        elif state.uploaded_image:
            st.image(state.uploaded_image)

        st.markdown("**Ingredient List:**")
        st.markdown("\n".join(map(lambda x: f'- {x}', state.recipe_ingredients)))

        st.markdown("**Steps:**")

        for step in state.recipe_steps:
            st.markdown(step.recipe_step)
            st.image(step.recipe_image_url, width=256)

        if st.button('Restart'):
            model.on_return_to_start()


def display_waiting_input():
    with input_holder.container():
        # Create a form
        with st.form(key='my_form'):
            img_file_buffer = st.camera_input(label="Take an ingredients picture")
            img_upload_buffer = st.file_uploader(label="Or upload an image")
            user_prompt = st.text_area(
                label="Based on the ingredients from the picture tell me what you want for your recipe"
            )

            # Add a submit button
            submit_button = st.form_submit_button(label='Submit')

        # Check if the form was submitted
        if submit_button:
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
