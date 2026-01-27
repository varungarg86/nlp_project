# Import necessary libraries

import json
import os
from pathlib import Path
import streamlit as st
from openai import OpenAI

# Define your API key in JSON Format
API_KEY_FILE = "api_key.json"

# Define your LLM Model
MODEL = "gpt-4o-mini"

# System Instructions for the LLM Prompt
SYSTEM = "You are a user friendly, helpful, and friendly AI assistant."
# Content of the JSON KEY FILE

# {
#     "OPENAI_API_KEY": "XXXX"
# }


# Define a function that reads the API key from a JSON format file
def load_api_key(path: str) -> str:
    p = Path(path)
    # Read the entire file text(UTF-8_ and parse JSON into a Python dict
    data = json.loads(p.read_text(encoding="utf-8"))
    key = data.get("OPENAI_API_KEY", "").strip()
    return key


# ------------- App Setup --------------
st.set_page_config(page_title="Demo Chatbot")

# Setting env var which makes API key to pickup automatically
os.environ["OPENAI_API_KEY"] = load_api_key(API_KEY_FILE)

# Initialize OpenAI client Instance
client = OpenAI()

st.title("Demo Chatbot")

# ----- Chat Memory using Streamlit session state -----

# While loading UI First time

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM}]

# While render Chat UI subsequent timings
# Loop through all saved messages so that user can see the past chat content
for m in st.session_state.messages:
    # Skip displaying system messages because it contains only initial instructions
    if m["role"] == "system":
        continue

    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ---- Accept new user Input ----------

prompt = st.chat_input("Type your message...")

if prompt:
    # Add the user message into session history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Render the user's latest message in the chat UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create an assistant chat container for the model response
    with st.chat_message("assistant"):
        # Just to show a spinner message
        with st.spinner("Analyzing..."):
            # Call OpenAI API to generate assistant reply
            resp = client.responses.create(
                model = MODEL,
                instructions=SYSTEM,
                input=[m for m in st.session_state.messages if m["role"] != "system"]
            )
            answer = (resp.output_text or "").strip() or "(No Output)"

            st.markdown(answer)

    # Save chat assistant message into session history
    st.session_state.messages.append({"role": "assistant", "content": answer})
