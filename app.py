import streamlit as st
from langchain_community.vectorstores import Chroma
import json
import google.generativeai as genai
import re
from langchain_huggingface import HuggingFaceEmbeddings
import helper_functions as hf


# === CONFIG ===

with open("api_google.txt") as f:
    api_key = json.load(f)["key"]

genai.configure(api_key=api_key)

# === INIT CHROMA ===
# client = chromadb.HttpClient(host='localhost', port=7000, settings=Settings(allow_reset=True))
# client = chromadb.PersistentClient(path="./chroma_RAG")
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2", model_kwargs={"device": "cpu"}
)
# collection = client.get_or_create_collection(name="ReproRAG")
# db = Chroma(
#     client=client,
#     collection_name="ReproRAG",
#     embedding_function=embedding_function,
# )

db = Chroma(
    collection_name="ReproRAG",
    persist_directory="./chromaRepro",
    embedding_function=embedding_function,
)
# Streamlit #

st.set_page_config(layout="wide", page_title="GSRM ChatBot")
st.title("GSRM supporter")

st.sidebar.header("Settings")

# Configurable chat history length
history_length = st.sidebar.number_input(
    "Max chat history exchanges to keep", min_value=1, max_value=20, value=5, step=1
)
retrieval_k = st.sidebar.slider(
    "Number of retrieved documents (k)", min_value=1, max_value=10, value=4, step=1
)
model_temperature = st.sidebar.slider(
    "Model Temperature (0.0 - 1.0)",
    min_value=0.0,
    max_value=1.0,
    value=0.8,  # Default value
    step=0.1,
    help="Lower values make the output more deterministic. Higher values lead to more creative.",
)
show_context = st.sidebar.checkbox("Show Retrieved Context", value=False)

model_name = st.sidebar.selectbox("Choose your model:", ("gemini-2.0-flash","gemma-3-27b-it", "gemini-2.5-flash"))

# Store chat history

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Reproductive field is messy and hard. Please, make you a favor and ask me")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            # Step 1: Retrieve context
            context_docs = hf.retrieve_context(query, k=retrieval_k, database=db)
            if show_context:
                st.markdown("### Retrievec Context:")
                st.markdown(context_docs)

            if st.session_state.chat_history:
                history_context = hf.format_chat_history(st.session_state.chat_history)
                context_docs = f"--- Chat History ---\n{history_context}\n--- RAG Context ---\n{context_docs}"

            # Step 2: Build prompt
            system_message = hf.get_system_message_rag(context_docs)
            full_prompt = hf.get_prompt(query, context=system_message)

            # Step 3: Run Gemini with streaming
            # model = genai.GenerativeModel(model_name="gemma-3-27b-it")
            # response = model.generate_content(
            #     full_prompt, stream=True, generation_config={"temperature": model_temperature}
            # )

            model = genai.GenerativeModel(model_name="gemma-3-27b-it")
            response = model.generate_content(
                full_prompt,
                stream=False,
                generation_config={"temperature": model_temperature},
            )

            # Step 4: Display and store output
            output = ""
            for chunk in response:
                # st.write(chunk.text, end="", unsafe_allow_html=True)
                output += chunk.text
            st.markdown(output, unsafe_allow_html=True)

            st.session_state.chat_history.append((query, output))
            st.session_state.chat_history = st.session_state.chat_history[
                -history_length:
            ]


# Display history
if st.session_state.chat_history:
    st.markdown("### Chat History")
    for idx, (q, a) in enumerate(st.session_state.chat_history, 1):
        with st.expander(f"Exchange {idx}: {q}"):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
