from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st

load_dotenv()


st.markdown('<h1 style="font-size:28px; margin:0;">💬 Generative AI Chat Application - By Venu</h1>', unsafe_allow_html=True)
st.set_page_config(
    page_title="Chat Application",
    page_icon="🤖",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    response = llm.invoke([{"role": "system", "content": "You are a helpful assistant."}, *st.session_state.messages])

    st.session_state.messages.append({"role": "assistant", "content": response.text})

    with st.chat_message("assistant"):
        st.markdown(response.text)
