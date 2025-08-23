# Libraries
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
import json
import uuid
import unicodedata
import requests
from tqdm import tqdm


# Functions


def clean_text(text):
    # Normalize weird Unicode characters to their closest ASCII equivalent
    text = unicodedata.normalize("NFKC", text)
    # Replace non-breaking hyphens and dashes with ASCII hyphen
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    # Remove form feed characters (\x0c), common page breaks
    text = text.replace("\x0c", " ")
    # Replace non-breaking spaces (\xa0) with regular space
    text = text.replace("\xa0", " ")
    # 3. Fix hyphenated line breaks (e.g., treat-\nment -> treatment)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Remove numbered citations like (1), (9, 10), (5–7), etc.
    text = re.sub(r"\(\s?\d+(?:\s?(?:,|-)\s?\d+)*\s?\)", "", text)
    # Remove mid-sentence line breaks: "word\nword" -> "word word"
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Replace multiple newlines with just two (preserve paragraphs)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


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
        7. When you provide information, you must also provide the reference of the article.
        8. Do not look up on internet.
        9. Only show your final response! 
        
        Constraints:
        1. DO NOT PROVIDE ANY EXPLANATION OR DETAILS OR MENTION THAT YOU WERE GIVEN CONTEXT. Only do that when questions are related to coding.
        2. Don't mention that you are not able to find the answer in the provided context.
        3. Ignore the part of the content that only contains references.
        3. Don't make up the answers by yourself.
        4. Try your best to provide answer from the given context.

        CONTENT:
        {content}
        """


def get_ques_response_prompt(question, context):
    return f"""
    Context\n:
    {context}
    ==============================================================
    Based on the above context, please provide the answer to the following question\n:
    {question}
    """


def get_pmid_from_doi(doi, api_key=None):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": f"{doi}[DOI]", "retmode": "json"}
    if api_key:
        params["api_key"] = api_key

    response = requests.get(url, params=params)
    result = response.json()
    pmids = result.get("esearchresult", {}).get("idlist", [])
    return pmids[0] if pmids else "None"


def get_pmcid_from_pmid(pmid, api_key=None):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json"}
    if api_key:
        params["api_key"] = api_key

    response = requests.get(url, params=params)

    try:
        data = response.json()
        pmcid_link = None
        linksets = data["linksets"][0]["linksetdbs"]

        for j in linksets:
            if j.get("linkname") == "pubmed_pmc":
                pmcid_link = j["links"][0]
                return pmcid_link
    except Exception:
        return None


def clean_doi(doi):
    """Remove prefixes from DOI links and standardize format."""
    return re.sub("(https://doi\.org/|http://dx\.doi\.org/)", "", doi.strip())


def load_embeddings(documents, chunk_size: int, embedding_model):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10,
        separators=["\n\n", "\n", ".", "!", "?", " "],  # smart splitting
    )

    info_splitted = []

    for j in documents:
        for key, value in j.items():
            if (
                key
                in [
                    "Abstract",
                    "Introduction",
                    "Methods",
                    "Results",
                    "Discussion",
                    "Conclusion",
                ]
                and value
            ):
                if len(value) > 1200:
                    chunks = splitter.split_text(value)

                    for i, c in enumerate(chunks):
                        info_splitted.append(
                            {
                                "chunk_index": i,
                                "content": j.get("Authors").split(",")[0]
                                + " et al.,"
                                + j.get("Publication", "Not identified")
                                + ", DOI:"
                                + j.get("DOI")
                                + "\n"
                                + c,
                                "parent": key,
                                "split": True,
                                "DOI": j.get("DOI"),
                                "Reference": j.get("Authors").split(",")[0]
                                + " et al.,"
                                + j.get("Publication", "Not identified"),
                            }
                        )
                else:
                    info_splitted.append(
                        {
                            "chunk_index": 0,
                            "content": j.get("Authors").split(",")[0]
                            + " et al.,"
                            + j.get("Publication", "Not identified")
                            + ", DOI:"
                            + j.get("DOI")
                            + "\n"
                            + value,
                            "parent": key,
                            "split": False,
                            "DOI": j.get("DOI"),
                            "Reference": j.get("Authors").split(",")[0]
                            + " et al.,"
                            + j.get("Publication", "Not identified"),
                        }
                    )

    texts = [chunk["content"] for chunk in info_splitted]
    metadatas = [
        {
            "parent": chunk["parent"],
            "chunk_index": chunk["chunk_index"],
            "DOI": chunk["DOI"],
            "Reference": chunk["Reference"],
        }
        for chunk in info_splitted
    ]
    ids = [str(uuid.uuid1()) for _ in metadatas]

    db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        ids=ids,
    )

    return db


def clean_doi_links(text):
    """
    Replace problematic Unicode dashes (like non-breaking hyphen) with normal ASCII dashes.
    """
    return re.sub(r"[\u2010-\u2015\u2212]", "-", text)


def retrieve_context(question, k, database):
    results = database.similarity_search(question, k)
    selected_index = []
    ideal_chunks = []
    meta_selected = []

    def is_new_chunk(r, selected_index):
        next_chunk = "_".join([r["parent"], r["Reference"], str(r["chunk_index"] + 1)])
        prev_chunk = "_".join([r["parent"], r["Reference"], str(r["chunk_index"] - 1)])
        return next_chunk not in selected_index and prev_chunk not in selected_index

    for doc in results:
        r = doc.metadata

        if r["parent"] not in ["Journal", "DOI"] and is_new_chunk(r, selected_index):
            ii = "_".join([r["parent"], r["Reference"], str(r["chunk_index"])])
            selected_index.append(ii)

            candidates = database.get(
                where={"$and": [{"Reference": r["Reference"]}, {"parent": r["parent"]}]}
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
        for text in ideal_chunks:
            context.append(f"Summary:\n\n{''.join(text)}\n\n")

    return context


def answer_with_rag(
    question: str,
    llm,
    database,
    template,
    num_docs_final=7,
    recursive_chunk=False,
):
    """Answer a question using RAG with the given knowledge index."""
    # Gather documents with retriever

    if recursive_chunk:
        relevant_docs = retrieve_context(
            question=question, database=database, k=num_docs_final
        )
    else:
        relevant_docs = database.similarity_search(query=question, k=num_docs_final)
        relevant_docs = [
            doc.page_content for doc in relevant_docs
        ]  # keep only the text

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
    )

    final_prompt = template.format(question=question, context=context)

    # Redact an answer
    answer = llm.invoke(final_prompt)

    return answer.content, relevant_docs


def run_rag_tests(
    eval_dataset,
    llm,
    database,
    output_file,
    recursive_chunk,
    verbose=False,
    test_settings=None,
    num_docs_final=7,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = answer_with_rag(
            question=question,
            llm=llm,
            database=database,
            recursive_chunk=recursive_chunk,
            num_docs_final=num_docs_final,
        )

        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"True answer: {example['answer']}")
        result = {
            "question": question,
            "true_answer": example["answer"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)


def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluator_name: str,
    evaluation_prompt_template,
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )

        eval_result = eval_chat_model.invoke(eval_prompt)
        feedback, score = [
            item.strip() for item in eval_result.content.split("[RESULT]")
        ]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f)


def correctness(inputs: dict, grader_llm,correctness_instructions):
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
    QUESTION: {inputs['question']}
    GROUND TRUTH ANSWER: {inputs['true_answer']}
    STUDENT ANSWER: {inputs['generated_answer']}"""

    # Run evaluator
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions}, 
        {"role": "user", "content": answers}
    ])
    
    return grade["correct"]

def groundedness(inputs: dict, grounded_llm, grounded_instructions):
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "\n\n".join(doc for doc in inputs["retrieved_docs"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {inputs['generated_answer']}"
    grade = grounded_llm.invoke([{"role": "system", "content": grounded_instructions}, {"role": "user", "content": answer}])
    return grade["grounded"]


def retrieval_relevance(inputs: dict, retrieval_llm, retrieval_relevance_instructions):
    """An evaluator for document relevance"""
    doc_string = "\n\n".join(doc for doc in inputs["retrieved_docs"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"

    grade = retrieval_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions}, 
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]

def relevance(inputs: dict, relevance_llm, relevance_instructions ) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {inputs['generated_answer']}"
    grade = relevance_llm .invoke([
        {"role": "system", "content": relevance_instructions}, 
        {"role": "user", "content": answer}
    ])
    
    return grade['relevant']


if __name__ == "__main__":
    print("hello world")
