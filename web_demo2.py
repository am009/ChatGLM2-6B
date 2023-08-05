from transformers import AutoModel, AutoTokenizer
import streamlit as st
from streamlit_chat import message
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import sys

st.set_page_config(
    page_title="ChatGLM2-6b",
    page_icon=":robot:",
    layout='wide'
)


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def stream_chat(query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192, top_p=0.8, temperature=0.8):
    import openai
    openai.api_base = "http://localhost:8086/v1"
    openai.api_key = "none"
    messages = []
    for q, response in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": query})
    print(f'messages: {messages}')

    response = ''
    for chunk in openai.ChatCompletion.create(
        model="chatglm2-6b",
        messages=messages,
        stream=True
    ):
        if hasattr(chunk.choices[0].delta, "content"):
            response += chunk.choices[0].delta.content
            new_history = history + [(query, response)]
            yield response, new_history


def predict(input, max_length, top_p, temperature, history=None):
    # tokenizer, model = get_model()
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            if len(history)>MAX_BOXES:
                history = history[-MAX_TURNS:]
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            for response, history in stream_chat(input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
                query, response = history[-1]
                st.write(response)

    return history


container = st.container()


if 'state' not in st.session_state:
    st.session_state['state'] = []

# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")

max_length = st.sidebar.slider(
    'max_length', 0, 32768, 8192, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.8, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.95, step=0.01
)


if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])
else:
    history = st.session_state['state']
    with container:
        if len(history) > 0:
            if len(history)>MAX_BOXES:
                history = history[-MAX_TURNS:]
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))
