import streamlit as st
import random
import time
from langchain_community.vectorstores import Chroma
import uuid
import chromadb
from chromadb.config import Settings
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


# Streamlit #

st.set_page_config(layout="wide")
st.title("Reproductive Chat")

st.sidebar.header("Settings")


# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # React to user input
# if prompt := st.chat_input("What is up?"):
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})


# def response_generator():
#     response = random.choice(
#         [
#             "Hello there! How can I assist you today?",
#             "Hi, human! Is there anything I can help you with?",
#             "Do you need help?",
#         ]
#     )

#     for word in response.split():
#         yield word + " "
#         time.sleep(0.05)

# with st.chat_message("assistant"):
#     response = st.write_stream(response_generator())



# with st.form("my_form"):
#     text = st.text_area(
#         "Enter text:",
#         "What are the three key pieces of advice for learning to code?"
#     )
#     submitted = st.form_submit_button("Submit")
#     generate_reponse(text)


# # Add assistant response to chat history
# st.session_state.messages.append({"role": "assistant", "content": response})

# Vector database



# def query_database(query_text):
#     # Prepare the DB

#     client = chromadb.HttpClient(host='localhost', port=8000,settings=Settings(allow_reset=True))
#     embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#     db = Chroma(
#     client=client,
#     collection_name="tfm",
#     embedding_function=embedding_function,
#     )



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
        6. Only show your final response! 
        
        Constraints:
        1. DO NOT PROVIDE ANY EXPLANATION OR DETAILS OR MENTION THAT YOU WERE GIVEN CONTEXT. Only do that when questions are related to coding.
        2. Don't mention that you are not able to find the answer in the provided context.
        3. Ignore the part of the content that only contains references.
        3. Don't make up the answers by yourself.
        4. Try your best to provide answer from the given context.

        CONTENT:
        {content}
        """

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


# LangChain #
llm = ChatOllama(model="gemma3:12b", streaming=True)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

prompt = PromptTemplate(
    template = system_promt,
    input_variables = ["context", "question"]
    )

# Chroma #
client = chromadb.HttpClient(host='localhost', port=8000,settings=Settings(allow_reset=True))
db = Chroma(
    client=client,
    collection_name="tfm",
    embedding_function=embedding_function,
    )


rag_chain = (
    {"context":db.as_retriver(), "question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def generate_response(input, rag_chain):
    result = rag_chain.invoke(input)
    return result


def retrieve(db, question):
     retrieve_documents = db.similarity_search(question)
     docs_content ="\n\n".join(doc.page_content for doc in retrieve_documents)
     return docs_content


# retriever = db.as_retriever(search_type="similarity")
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Session State Setup #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Display Chat History #
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input #

if prompt := st.chat_input("Say something"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = st.empty()

        # Retrieve information

        
        full_response = llm.invoke({"question": prompt ,"context" : get_system_message_rag(retrieve(db,prompt))})

        # retrieved_docs = retriever.get_relevant_documents(prompt)
        # full_response = (
        #     "No relevant documents found." if not retrieved_docs
        #     else qa({"query": prompt}).get("result", "No response generated.")
        # )

        response.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
