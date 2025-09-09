import streamlit as st
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
import os, time
from datapipeline import wiki_search, arxiv_search, retrieve
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv

st.title("🤖💡📚 Multi AI RAG Chatbot")

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.environ["GROQ_API_KEY"]

## LLM Model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search", "arxiv_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router

user_prompt = st.text_input("Enter your question")

## Edges
def route_question(state):
    """
    Route question to wiki search or arxiv search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        return "wiki_search"
    elif source.datasource == "arxiv_search":
        return "arxiv_search"
    elif source.datasource == "vectorstore":
        return "vectorstore"
    

workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("wiki_search", wiki_search)  # web search
workflow.add_node("arxiv_search", arxiv_search)  # arxiv search
workflow.add_node("retrieve", retrieve)  # retrieve

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "arxiv_search": "arxiv_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
workflow.add_edge("arxiv_search", END)
# Compile
app = workflow.compile()

if user_prompt:
    start = time.process_time()

    inputs = {"question": user_prompt}
    final_documents = None

    for output in app.stream(inputs):
        for key, value in output.items():
            st.write(f"🔹 Node `{key}` processed")
            final_documents = value.get("documents", None)
            if value.get("documents"):
                if key == "retrieve":
                    st.write("Fetching from Vector Store DB")
                    st.write("Output:")
                    st.write(value["documents"][0].dict()["metadata"]["description"])
                else:
                    st.write("✅ Output:")
                    st.write(value["documents"].dict()["page_content"])

    st.write("⏱️ Response time:", time.process_time() - start)


