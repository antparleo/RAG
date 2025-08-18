import streamlit as st
import chromadb

# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
import json
import google.generativeai as genai
import re
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings


# === CONFIG ===
with open("api_google.txt") as f:
    api_key = json.load(f)["key"]

genai.configure(api_key=api_key)

# === INIT CHROMA ===
# client = chromadb.HttpClient(host='localhost', port=7000, settings=Settings(allow_reset=True))
# client = chromadb.PersistentClient(path="./chroma_RAG")
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="avsolatorio/GIST-small-Embedding-v0")
# collection = client.get_or_create_collection(name="ReproRAG")
# db = Chroma(
#     client=client,
#     collection_name="ReproRAG",
#     embedding_function=embedding_function,
# )

db = Chroma(
    collection_name="ReproRAG",
    persist_directory="./chromaRepro",
    embedding_function = embedding_function
)


# === RAG UTILS ===


def clean_doi_links(text):
    """
    Replace problematic Unicode dashes (like non-breaking hyphen) with normal ASCII dashes.
    """
    return re.sub(r"[\u2010-\u2015\u2212]", "-", text)


def retrieve_context(question, k=4):
    """
    Retrieves relevant context documents from the Chroma vector database based on the input question.

    Parameters:
        question (str): The user's question to search for similar documents.
        k (int, optional): The number of top similar documents to retrieve. Defaults to 4.

    Returns:
        str: A formatted string containing the relevant context extracted from the documents.
    """
    results = db.similarity_search(question, k)
    selected_index = []
    ideal_chunks = []
    meta_selected = []

    def is_new_chunk(r, selected_index):
        next_chunk = "_".join([r["parent"], r["Reference"], str(r["chunk_index"] + 1)])
        prev_chunk = "_".join([r["parent"], r["Reference"], str(r["chunk_index"] - 1)])
        return next_chunk not in selected_index and prev_chunk not in selected_index

    for doc in results:
        r = doc.metadata

        if r["parent"] not in ["Journal", "URL"] and is_new_chunk(r, selected_index):
            ii = "_".join([r["parent"], r["Reference"], str(r["chunk_index"])])
            selected_index.append(ii)

            candidates = db.get(
                where={
                    "$and": [{"Reference": r["Reference"]}, {"parent": r["parent"]}]
                }
            )

            max_index = len(candidates["metadatas"]) - 1

            meta_selected.append(candidates["metadatas"])
            ideal_chunks.append(
                [
                    doc
                    for doc, meta in zip(
                        candidates["documents"], candidates["metadatas"]
                    )
                    if meta["chunk_index"]
                    in [
                        r["chunk_index"],
                        max(r["chunk_index"] - 1, 0),
                        min(r["chunk_index"] + 1, max_index),
                    ]
                ]
            )

    context = []
    for text, meta in zip(ideal_chunks, meta_selected):
        if meta:  # Only proceed if meta is not empty
            doi = clean_doi_links(meta[0]["DOI"]) if "DOI" in meta[0] else "DOI not available"
            context.append(
                f"Reference:{meta[0]['Reference']}\n\nLink (DOI)\n: {doi}\n\nSummary:\n\n{''.join(text)}\n\n"
            )

    return "\n\n".join(context)


def get_system_message_rag(content):
    """
    Constructs a system message prompt for the RAG chatbot, guiding the model to answer questions using provided scientific context.

    Parameters:
        content (str): The context information extracted from relevant documents.

    Returns:
        str: A formatted system message prompt for the language model.
    """
    return f"""You are an expert consultant helping executive advisors to get relevant information from scientific articles and code related to reproduction and bioinformatics.

Generate your response by following the steps below:
1. Recursively break down the question into smaller questions to better understand it.
2. For each question/directive:
   2a. Select the most relevant information from the context in light of the conversation history.
3. Generate a draft response using selected information.
4. Remove duplicate content from draft response.
5. Generate Final Response with Comprehensive Referencing:
    5a. Provide the final, refined scientific text.
    5b. For every piece of information, data, or concept derived from the provided context, include a precise reference (Author, Journal, Year).
    5c. Ensure each unique source is cited only once at the end of the response in a consolidated reference list, unless distinct information from the same source is presented in different contexts requiring separate in-text citations for clarity.
6. Do not try to summarize the answers, explain it properly.
7. When you have provided all information, you must also give the references (author, Journal, Year) of the article and include here the DOI at the end.
8. Do not look up on internet.
9. Only show your final response! Don't show how you break down the question, only the final response.

Constraints:
- Don't mention that you are not able to find the answer in the provided context.
- Ignore the part of the content that only contains references.
- Don't make up the answers by yourself. If the information is not included in the context, say it.
- Try your best to provide answer from the given context.

CONTENT:
{content}
"""


# - DO NOT PROVIDE ANY EXPLANATION OR DETAILS OR MENTION THAT YOU WERE GIVEN CONTEXT.


def get_prompt(question, context):
    """
    Constructs a prompt for the language model by combining the provided context and question.

    Parameters:
        question (str): The user's question to be answered.
        context (str): The relevant context information extracted from documents.

    Returns:
        str: A formatted prompt string for the language model.
    """
    return f"""
Context:
{context}
==============================================================
Based on the above context, please provide the answer to the following question:
{question}
"""

def format_chat_history(chat_history):
    formatted = ""
    for i, (question, answer) in enumerate(chat_history):
        formatted += f"Previous Question {i + 1}: {question}\nAnswer: {answer}\n\n"
    return formatted


# Streamlit #

st.set_page_config(layout="wide", page_title="GSRM ChatBot")
st.title("GSRM ChatBot")

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
    value=0.8, # Default value
    step=0.1,
    help="Lower values make the output more deterministic. Higher values lead to more creative."
)
show_context = st.sidebar.checkbox("Show Retrieved Context", value=False)

# Store chat history

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Type your scientific question here")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            # Step 1: Retrieve context
            context_docs = retrieve_context(query, k=retrieval_k)
            if show_context:
                st.markdown("### Retrievec Context:")
                st.markdown(context_docs)

            if st.session_state.chat_history:
                history_context = format_chat_history(st.session_state.chat_history)
                context_docs = f"--- Chat History ---\n{history_context}\n--- RAG Context ---\n{context_docs}"

            # Step 2: Build prompt
            system_message = get_system_message_rag(context_docs)
            full_prompt = get_prompt(query, context=system_message)

            # Step 3: Run Gemini with streaming
            # model = genai.GenerativeModel(model_name="gemma-3-27b-it")
            # response = model.generate_content(
            #     full_prompt, stream=True, generation_config={"temperature": model_temperature}
            # )


            model = genai.GenerativeModel(model_name="gemma-3-27b-it")
            response = model.generate_content(
                full_prompt, stream=False, generation_config={"temperature": model_temperature}
            )

            # Step 4: Display and store output
            output = ""
            for chunk in response:
                # st.write(chunk.text, end="", unsafe_allow_html=True)
                output += chunk.text
            st.markdown(output, unsafe_allow_html=True)

            st.session_state.chat_history.append((query, output))
            st.session_state.chat_history = st.session_state.chat_history[-history_length:]


# Display history
if st.session_state.chat_history:
    st.markdown("### Chat History")
    for idx, (q, a) in enumerate(st.session_state.chat_history, 1):
        with st.expander(f"Exchange {idx}: {q}"):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
