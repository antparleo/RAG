# Dependencies

import langchain_community
import langchain_text_splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import streamlit as st

# Vector database

import uuid
import chromadb
from chromadb.config import Settings

# Load documents #

# load the document and split it into pages
loader = PyPDFLoader("/data/local/aparraga/Bioinformatician/RAG/Publications/Parraga-Leo2023.pdf")
pages = loader.load_and_split()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
storage = text_splitter.split_documents(pages)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# Vectorize content #

client = chromadb.HttpClient(host='localhost', port=8000, settings=Settings(allow_reset=True))

# Retrieve collection previusly created

db = Chroma(
    client=client,
    collection_name="tfm",
    embedding_function=embedding_function,
)


# Initializer #
llm = ChatOllama(model="gemma3:12b", streaming=True)



# Prompt #

system_promt = """You are an expert consultant helping executive advisors to get relevant information from scientific articles and code related to reproduction and bioinformatics.

        Generate your response by following the steps below:
        1. Recursively break down the question into smaller questions to better understand it.
        2. For each question/directive:
            2a. Select the most relevant information from the context in light of the conversation history.
        3. Generate a draft response using selected information.
        4. Remove duplicate content from draft response.
        5. Generate your final response after adjusting it to increase accuracy and relevance.
        6. Do not try to summarize the answers, explain it properly.
        6. Only show your final response! 
        
        Constraints:
        1. DO NOT PROVIDE ANY EXPLANATION OR DETAILS OR MENTION THAT YOU WERE GIVEN CONTEXT. Only do that when questions are related to coding.
        2. Don't mention that you are not able to find the answer in the provided context.
        3. Ignore the part of the content that only contains references.
        3. Don't make up the answers by yourself.
        4. Try your best to provide answer from the given context.

        CONTENT:
        {content}

        Question:
        {question}
        """



prompt = PromptTemplate(
    template = system_promt,
    input_variables = ["content", "question"]
    )


# Create a chain # 

rag_chain = (
  {"content": db.as_retriever(search_type="similarity"),  "question": RunnablePassthrough()} 
  | prompt 
  | llm
  | StrOutputParser() 
)


# Outside ChatBot() class
# input = input("Ask me anything: ")
# result = rag_chain.invoke(input)
# print(result)

st.set_page_config(page_title="Random Fortune Telling Bot")
with st.sidebar:
    st.title('Random Fortune Telling Bot')

# Function for generating LLM response
def generate_response(input, rag_chain):
    result = rag_chain.invoke(input)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, ask whatever related to GSRM you want"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from mystery stuff.."):
            response = generate_response(input, rag_chain=rag_chain) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)