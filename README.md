README.txt
==========

Improving the onboarding of researchers in reproductive medicine and omics (RAG)
---------------------------------------------

This project implements a Retrieval Augmented Generation (RAG) pipeline to evaluate and visualize the performance of various models and configurations for answering questions related to the reproductive system. The project includes scripts for data preprocessing, evaluation, database creation, and visualization.

Folder Structure:
-----------------
- `00_exploratory.ipynb`: Initial exploratory analysis of the data. Loads and inspects datasets and retrieve all the important documents to create the context for the RAG.
- `01_evaluation.ipynb`: Evaluates the performance of the RAG pipeline using multiple metrics (Correctness, Relevance, Groundedness, Retrieval).
- `02_database_creation.ipynb`: Creates the ChromaDB database to store embeddings according to the best paramters obtained in the evaluation process.
- `03_plots.ipynb`: Generates visualizations for the evaluation results. Includes bar plots, violin plots, and line plots to analyze metrics such as Correctness, Relevance, Groundedness, and Retrieval.
- `app.py`: A Streamlit-based web application for interacting with the RAG pipeline. Allows users to query the system and visualize results dynamically.
- `helper_functions.py`: Contains important functions to develop this RAG pipeline, including evaluation metrics, prompt generation, and database interaction.

Key Features:
-------------
1. **Evaluation Metrics**:
   - Correctness: Measures the accuracy of generated answers when they are compared with the ground truth.
   - Relevance: Evaluates the helpfulness of answers.
   - Groundedness: Assesses whether answers are based on retrieved documents.
   - Retrieval: Analyzes the quality of document retrieval.

2. **Visualization**:
   - Bar plots for standard deviation and mean of metrics.
   - Violin plots for distribution of scores.
   - Line plots for metric trends across different configurations.

3. **Database Creation**:
   - Processes scientific articles and stores embeddings in a Chroma database.
   - Supports recursive chunking for efficient text splitting.

4. **Interactive Web App**:
   - Built with Streamlit for querying the RAG pipeline.
   - Configurable settings for chat history, number of retrieved documents, and model temperature.
