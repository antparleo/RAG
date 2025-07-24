import streamlit as st
import chromadb
from chromadb.config import Settings
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import Chroma
import json
import google.generativeai as genai
import re


# === CONFIG ===
with open('api_google.txt') as f:
    api_key = json.load(f)['key']

genai.configure(api_key=api_key)

# === INIT CHROMA ===
client = chromadb.PersistentClient(path="./chroma_RAG")
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="prueba")
db = Chroma(
    client=client,
    collection_name="prueba",
    embedding_function=embedding_function,
)


# === RAG UTILS ===

def clean_doi_links(text):
    """
    Replace problematic Unicode dashes (like non-breaking hyphen) with normal ASCII dashes.
    """
    return re.sub(r'[\u2010-\u2015\u2212]', '-', text)

def retrieve_context(question, k=4):

    results = db.similarity_search(question,k)

    # results = collection.query(
    #     query_texts=[question],
    #     n_results=k
    # )

    texts_re = []
    metadata_re = []

    for doc in results:
        texts_re.append(doc.page_content)
        metadata_re.append(doc.metadata)

    context = []
    for text, meta in zip(texts_re, metadata_re):
        doi = clean_doi_links(meta["DOI"])
        context.append(f'Reference:{meta["Reference"]}\n\nLink (DOI)\n: {doi}\n\nSummary:\n\n{text}\n\n')

    return "\n\n".join(context)


def get_system_message_rag(content):
    return f"""You are an expert consultant helping executive advisors to get relevant information from scientific articles and code related to reproduction and bioinformatics.

Generate your response by following the steps below:
1. Recursively break down the question into smaller questions to better understand it.
2. For each question/directive:
   2a. Select the most relevant information from the context in light of the conversation history.
3. Generate a draft response using selected information.
4. Remove duplicate content from draft response.
5. Generate your final response after adjusting it to increase accuracy and relevance.
6. Do not try to summarize the answers, explain it properly.
7. When you provide information, you must also provide the reference (author, Journal, Year) of the article and its DOI. But only once to avoid being redudant unless the information coming from different scientific articles. Add only once the reference at the end of your response.
8. Do not look up on internet.
9. Only show your final response! 

Constraints:
- DO NOT PROVIDE ANY EXPLANATION OR DETAILS OR MENTION THAT YOU WERE GIVEN CONTEXT.
- Don't mention that you are not able to find the answer in the provided context.
- Ignore the part of the content that only contains references.
- Don't make up the answers by yourself.
- Try your best to provide answer from the given context.

CONTENT:
{content}
"""


def get_prompt(question, context):
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
        formatted += f"Previous Question {i+1}: {question}\nAnswer: {answer}\n\n"
    return formatted


# Streamlit #

st.set_page_config(layout="wide", page_title="GSRM ChatBot")
st.title("GSRM ChatBot")

st.sidebar.header("Settings")

# Store chat history

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your question")

if st.button("Ask"):
    if query:

        with st.spinner("Thinking..."):

            # Step 1: Retrieve context
            context_docs = retrieve_context(query)
            st.write(context_docs)

            if st.session_state.chat_history:
                history_context = format_chat_history(st.session_state.chat_history)
                context_docs = f"--- Chat History ---\n{history_context}\n--- RAG Context ---\n{context_docs}"

            # Step 2: Build prompt
            system_message = get_system_message_rag(context_docs)
            full_prompt = get_prompt(query, context=system_message)

            # Step 3: Run Gemini with streaming
            model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            response = model.generate_content(full_prompt, stream=True,  generation_config={"temperature": 0.8})

            # Step 4: Display and store output
            st.markdown("### Answer")
            output = ""
            for chunk in response:
                #st.write(chunk.text, end="", unsafe_allow_html=True)
                output += chunk.text
            st.write(output,unsafe_allow_html=True)

            st.session_state.chat_history.append((query, output))


# Display history
if st.session_state.chat_history:
    st.markdown("### Chat History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")