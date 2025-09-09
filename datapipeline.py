from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.schema import Document
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.runnables import RunnableLambda
import cassio


ASTRA_DB_APPLICATION_TOKEN = ""
ASTRA_DB_ID = ""
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

## Vector store DB
if "retriever" not in st.session_state:
    st.session_state.retriever = None

def create_vector_embedding():
    if "vectors" not in st.session_state:
        # Embeddings
        st.session_state.embeddings = OllamaEmbeddings(
            model="mxbai-embed-large:latest"
        )

        # Data ingestion from web URLs
        urls = [
            "https://scikit-learn.org/stable/documentation.html",
            "https://pytorch.org/docs/stable/index.html",
            "https://pytorch.org/tutorials/",
            "https://www.tensorflow.org/learn",
            "https://www.tensorflow.org/api_docs",
            "https://keras.io/",
            "https://docs.fast.ai/",
            "https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/index.html",
            "https://onnx.ai/",
            "https://lightgbm.readthedocs.io/en/stable/",
            "https://xgboost.readthedocs.io/en/stable/",
            "https://catboost.ai/en/docs/",
            "https://optuna.readthedocs.io/en/stable/",
            "https://ray.io/docs/",
            "https://docs.ray.io/en/latest/tune/index.html",
            "https://mlflow.org/docs/latest/index.html",
            "https://docs.wandb.ai/",
            "https://hydra.cc/docs/intro/",
            "https://cleverhans.readthedocs.io/en/latest/",
            "https://spacy.io/usage",
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
            "https://huggingface.co/docs/transformers/index",
            "https://huggingface.co/docs/datasets/index",
            "https://huggingface.co/docs/tokenizers/index",
            "https://huggingface.co/docs/evaluate/index",
            "https://huggingface.co/docs/accelerate/index",
            "https://huggingface.co/docs/trl/index",
            "https://huggingface.co/docs/peft/index",
            "https://huggingface.co/docs/autotrain/index",
            "https://docs.scipy.org/doc/numpy/",
            "https://pandas.pydata.org/docs/",
            "https://matplotlib.org/stable/contents.html",
            "https://seaborn.pydata.org/",
            "https://bokeh.org/docs/",
            "https://plotly.com/python/",
            "https://cloud.google.com/vertex-ai/docs",
            "https://azure.microsoft.com/en-us/products/ai-services/",
            "https://learn.microsoft.com/en-us/azure/machine-learning/",
            "https://aws.amazon.com/sagemaker/",
            "https://aws.amazon.com/machine-learning/",
            "https://docs.oracle.com/en-us/iaas/Content/ai-ml/",
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
        ]
        st.session_state.loader = [WebBaseLoader(url).load() for url in urls]
        st.session_state.docs = [item for sublist in st.session_state.loader for item in sublist]  

        # Text splitting
        st.session_state.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=5000, chunk_overlap=300
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs
        )

        # Cassandra vectorstore DB
        st.session_state.vectors = Cassandra(
            embedding=st.session_state.embeddings,
            table_name="qa_mini_demo"
        )
        st.session_state.vectors.add_documents(st.session_state.final_documents)
        st.session_state.vector_index = VectorStoreIndexWrapper(
            vectorstore=st.session_state.vectors
        )
        st.session_state.retriever = RunnableLambda(
            lambda q: st.session_state.vector_index.vectorstore.as_retriever().get_relevant_documents(q)
        )

create_vector_embedding()

retriever = st.session_state.retriever

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

## Arxiv and wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=10000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=10000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

## wiki search
def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---wikipedia---")
    question = state["question"]
    print(question)

    # Wiki search
    docs = wiki.invoke({"query": question})
    # print(docs["summary"])
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}


## arxiv search
def arxiv_search(state):
    """
    arxiv search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---arxiv---")
    question = state["question"]
    print(question)

    # arxiv search
    docs = arxiv.invoke({"query": question})
    # print(docs["summary"])
    arxiv_results = docs
    arxiv_results = Document(page_content=arxiv_results)

    return {"documents": arxiv_results, "question": question}