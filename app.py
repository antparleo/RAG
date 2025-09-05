import streamlit as st
from langchain_community.vectorstores import Chroma
import json
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
import helper_functions as hf


# === CONFIG ===

with open("api_google.txt") as f:
    api_key = json.load(f)["key"]

genai.configure(api_key=api_key)


embedding_function = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2", model_kwargs={"device": "cuda"}
)

db = Chroma(
    collection_name="ReproRAG",
    persist_directory="./chromaRepro",
    embedding_function=embedding_function,
)

# Streamlit #

st.set_page_config(layout="wide", page_title="GSRM ChatBot")
st.title("Reproductive System Knowledge Assistant")

supporter, infogroup = st.tabs(["Assistant", "Research group info"])

# Images

with st.sidebar:
    c1, c2, c3 = st.columns([1, 2, 1])   # middle column is wider
    with c2:
        st.image("cover.jpeg", width=500)  # your file


st.sidebar.header("Settings")

# Configurable chat history length
history_length = st.sidebar.number_input(
    "Max chat history exchanges to keep", min_value=1, max_value=20, value=5, step=1
)
retrieval_k = st.sidebar.slider(
    "Number of retrieved documents (k)", min_value=1, max_value=10, value=8, step=1
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

model_name = st.sidebar.selectbox(
    "Choose your model:", ("gemini-2.0-flash", "gemma-3-27b-it", "gemini-2.5-flash")
)

with supporter:

    # Store chat history

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input(
        "Reproductive field is messy and hard. Please, make you a favor and ask me"
    )

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
                system_message = hf.get_system_message_rag_streamlit(context_docs)
                full_prompt = hf.get_prompt(query, context=system_message)

                # Step 3: Run Gemini with streaming
                # model = genai.GenerativeModel(model_name="gemma-3-27b-it")
                # response = model.generate_content(
                #     full_prompt, stream=True, generation_config={"temperature": model_temperature}
                # )

                model = genai.GenerativeModel(model_name=model_name)
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

# About us

members = [
    {"name": "Patricia Díaz Gimeno", "degree": "PhD in Biomedicine", "email": "patricia.diaz@ivirma.com", "image": "photos/patridiaz.jpg"},
    {"name": "Patricia Sebastian León", "degree": "PhD in Statistics", "email": "patricia.sebastian@ivirma.com", "image": "photos/patrisebas.jpg"},
    {"name": "Antonio Párraga Leo", "degree": "PhD in Biomedicine & MSc in Bioinformatics", "email": "antonio.parraga@ivirma.com", "image": "photos/Antonio.jpg"},
    {"name": "Francisco José Sanz López", "degree": "PhD in Biomedicine", "email": "francisco.sanz@ivirma.com", "image": "photos/Fran.jpg"},
    {"name": "Diana Martí García", "degree": "PhD in Biomedicine", "email": "diana.marti@ivirma.com", "image": "photos/Diana.jpg"},
    {"name": "Asunta Martinez Martinez", "degree": "MSc in Reproductive Medicine & Bioinformatics", "email": "asunta.martinez@ivirma.com", "image": "photos/Asunta.jpg"},
    {"name": "Nataly Del Aguila De Cárdenas", "degree": "MSc in Reproductive Medicine", "email": "nataly.cardenas@ivirma.com", "image": "photos/Nataly.jpeg"},
    {"name": "Rebeca Esteve Moreno", "degree": "MSc in Reproductive Medicine", "email": "rebeca.esteve@ivirma.com", "image": "photos/Rebeca.jpeg"},
    {"name": "María Salvaleda Mateu", "degree": "MSc in Reproductive Medicine", "email": "maria.salvaleda@ivirma.com", "image": "photos/Maria.jpeg"},
    {"name": "Elena Perez Rico", "degree": "MSc in Reproductive Medicine & Bioinformatics", "email": "elena.perezri@ivirma.com", "image": "photos/elena.jpg"},
]


with infogroup:

    n_cols = 3  # how many profiles per row
    for i in range(0, len(members), n_cols):
        cols = st.columns(n_cols)
        st.markdown("\n\n ----------------------------------------------------")
        for col, member in zip(cols, members[i:i+n_cols]):
            with col:
                st.image(member["image"], width=150)
                st.markdown(f"**{member['name']}**")
                st.markdown(member["degree"])
                st.markdown(f"[{member['email']}](mailto:{member['email']})")