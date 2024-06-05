import streamlit as st

from src.streamlit.session import SessionKeys
from src.streamlit.system import SystemModel, DisplayState

if SessionKeys.SYSTEM_MODEL not in st.session_state:
    st.session_state[SessionKeys.SYSTEM_MODEL] = SystemModel()

model: SystemModel = st.session_state[SessionKeys.SYSTEM_MODEL]


def display_state(state: DisplayState):
    with container:
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                state.image,
                caption="You just uploaded an image!",
            )

        with col2:
            if not state.recipe_text:
                st.markdown(state.loading_message)
            if not state.loading_message:
                st.markdown(state.recipe_text)


container = st.empty()
img_file_buffer = st.camera_input(label="Take an ingredients picture")
img_upload_buffer = st.file_uploader(label="Or upload an image")
user_prompt = st.text_area(label="Based on the ingredients from the picture tell me what you want for your recipe")

# # Define an asyncio scheduler
# loop = asyncio.get_event_loop()
# asyncio_scheduler = AsyncIOScheduler(loop)

# Observe events from the ViewModel
events = model.observe_events().pipe()

model.on_user_input(img_file_buffer, img_upload_buffer, user_prompt)

events.subscribe(
    on_next=lambda event: display_state(event),
    on_error=lambda e: print(f"Error: {e}"),
    on_completed=lambda: print("Stream completed")
)
