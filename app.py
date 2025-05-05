import streamlit as st
from typing import Generator
from groq import Groq

# ------------------------------------------------------------------------------
# SYSTEM PROMPT (DataEngineerGPT)
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are DataEng GPT, a world-class AI assistant with PhD-level expertise and over a decade of "
    "industry experience in Data Engineering, DevOps, and Cloud Architecture. Your mission is to help users design, "
    "implement, debug, and optimize robust, scalable data solutions—error-free and production-ready.\n"
    "\n"
    "Core Expertise:\n"
    "- Python (PySpark), SQL, Streamlit, Flask, Spark, Kafka, Airflow, ADF\n"
    "- Azure (Synapse, Data Lake, Databricks), AWS (Glue, Redshift, EMR), GCP (Dataflow, BigQuery)\n"
    "- Data modeling (Kimball/Inmon), CI/CD, Terraform, best practices for security, testing, and cost optimization.\n"
    "\n"
    "How You Operate:\n"
    "1. Ingest full chat history; refer back to previous code, preferences, & environment details.\n"
    "2. Break requests into phases: design, implementation, testing, optimization, monitoring.\n"
    "3. Provide clean, commented, PEP8-compliant code blocks with error handling and logging.\n"
    "4. Highlight performance, cost, and security considerations.\n"
    "5. Ask concise clarifying questions if any requirement or environment detail is missing.\n"
    "6. Maintain a professional yet approachable tone and suggest learning resources."
)

# ------------------------------------------------------------------------------
# Streamlit App Configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_icon="💬", layout="wide", page_title="DataEng GPT")

# Display application icon and title
st.markdown(
    """<div style="display: flex; align-items: center; margin-bottom: 1rem;">
         <span style="font-size: 78px; line-height: 1">🤖</span>
         <h1 style="margin-left: 1rem;">DataEng GPT</h1>
     </div>""",
    unsafe_allow_html=True,
)

# Initialize Groq client
client = Groq(api_key=st.secrets['GROQ_API_KEY'])

# Session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT}
    ]
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Updated Model definitions
models = {
    'meta-llama/llama-4-scout-17b-16e-instruct': {
        'name': 'LLaMA 4 Scout (17B-16E-Instruct)', 'tokens': 8192
    },
    'meta-llama/llama-4-maverick-17b-128e-instruct': {
        'name': 'LLaMA 4 Maverick (17B-128E-Instruct)', 'tokens': 8192
    },
    'qwen-qwq-32b': {
        'name': 'Qwen-QwQ-32B', 'tokens': 131072
    },
    'deepseek-r1-distill-llama-70b': {
        'name': 'DeepSeek R1 Distill LLaMA 70B', 'tokens': 131072
    },
    'compound-beta': {
        'name': 'Groq Compound Beta', 'tokens': 8192
    }
}

# Layout: model selector and token slider
col1, col2 = st.columns([2, 1])
with col1:
    model_option = st.selectbox(
        "Choose a model:", list(models.keys()),
        format_func=lambda k: models[k]['name'], index=0
    )
    if st.session_state.selected_model != model_option:
        # Reset messages but keep system prompt
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.selected_model = model_option
with col2:
    max_tokens = st.slider(
        "Max Tokens:", 512, models[model_option]['tokens'],
        value=min(8192, models[model_option]['tokens']), step=512,
        help=f"Max tokens per response (up to {models[model_option]['tokens']})."
    )

# Display chat history (skip system messages)
for msg in st.session_state.messages:
    if msg['role'] == 'system':
        continue
    avatar = '🤖' if msg['role'] == 'assistant' else '👨‍💻'
    with st.chat_message(msg['role'], avatar=avatar):
        st.markdown(msg['content'])

# Helper to stream responses
def generate_responses(stream) -> Generator[str, None, None]:
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# User input
if prompt := st.chat_input("Enter your data engineering question or request..."):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.chat_message('user', avatar='👨‍💻').markdown(prompt)

    payload = [
        {'role': m['role'], 'content': m['content']}
        for m in st.session_state.messages
    ]
    try:
        response_stream = client.chat.completions.create(
            model=model_option,
            messages=payload,
            max_tokens=max_tokens,
            stream=True,
        )
        # Stream and render assistant's reply correctly
        with st.chat_message('assistant', avatar='🤖'):
            full_reply = st.write_stream(generate_responses(response_stream))
        st.session_state.messages.append({'role': 'assistant', 'content': full_reply})
    except Exception as e:
        st.error(f"API Error: {e}", icon="🚨")

