import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

st.set_page_config(
    page_title="Chat Application",
    page_icon="🤖",
    layout="centered",
)

load_dotenv()

# Model configuration
MODEL_CONFIG = {
    "Groq - qwen/qwen3-32b": {
        "provider": "groq",
        "model_name": "qwen/qwen3-32b",
        "api_key_env": "GROQ_API_KEY"
    },
    "Groq - llama-3.3-70b-versatile": {
        "provider": "groq",
        "model_name": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY"
    },
    "Groq - llama-3.1-8b-instant": {
        "provider": "groq",
        "model_name": "llama-3.1-8b-instant",
        "api_key_env": "GROQ_API_KEY"
    },
    "Groq - openai/gpt-oss-120b": {
            "provider": "groq",
            "model_name": "openai/gpt-oss-120b",
            "api_key_env": "GROQ_API_KEY"
        },
    "Groq - openai/gpt-oss-20b": {
        "provider": "groq",
        "model_name": "openai/gpt-oss-20b",
        "api_key_env": "GROQ_API_KEY"
    },
    "Gemini - gemini-2.5-flash": {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "api_key_env": "GOOGLE_API_KEY"
    },
}

st.markdown('<h1 style="font-size:28px; margin:0;">💬 Generative AI Chat Application - By Venu</h1>', unsafe_allow_html=True)

# Sidebar for model selection
with st.sidebar:
    st.header("⚙️ Settings")

    selected_model = st.selectbox(
        "Select AI Model",
        options=list(MODEL_CONFIG.keys()),
        index=0,
        help="Choose the AI model for your conversation"
    )

    model_info = MODEL_CONFIG[selected_model]

    # Display model information
    st.info(f"**Provider:** {model_info['provider'].title()}\n\n**Model:** {model_info['model_name']}")

    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Higher values make output more creative, lower values more focused"
    )

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # API Key status
    api_key = os.getenv(model_info['api_key_env'])
    if api_key:
        st.success(f"✅ {model_info['api_key_env']} configured")
    else:
        st.error(f"❌ {model_info['api_key_env']} not found")
        st.warning("Please add your API key to the .env file")

# Initialize session state

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize LLM based on selected model
try:
    if model_info['provider'] == "groq":
        llm = ChatGroq(
            model=model_info['model_name'],
            temperature=temperature,
            api_key=os.getenv(model_info['api_key_env'])
        )
    elif model_info['provider'] == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=model_info['model_name'],
            temperature=temperature,
            google_api_key=os.getenv(model_info['api_key_env'])
        )
    else:
        st.error(f"Unsupported provider: {model_info['provider']}")
        st.stop()
except Exception as e:
    st.error(f"Error initializing model: {str(e)}")
    st.stop()

user_input = st.chat_input("Type your message here...")

if user_input:
    # Check if API key is available
    if not os.getenv(model_info['api_key_env']):
        st.error(f"❌ {model_info['api_key_env']} not configured. Please add it to your .env file.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {selected_model}..."):
                # Prepare messages for the model
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                messages.extend(st.session_state.messages)

                # Get response from LLM
                response = llm.invoke(messages)

                # Extract content from response (both providers return content attribute)
                response_content = response.content if hasattr(response, 'content') else str(response)

                st.markdown(response_content)

        st.session_state.messages.append({"role": "assistant", "content": response_content})

    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        # Remove the user message if there was an error
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.pop()
