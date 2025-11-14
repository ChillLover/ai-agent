from operator import add

import pandas as pd

from phoenix.client import Client
from phoenix.evals import ClassificationEvaluator, async_evaluate_dataframe
from phoenix.evals.llm import LLM
from phoenix.evals.utils import to_annotation_dataframe
from phoenix.otel import register

from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool, ToolRuntime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from dotenv import load_dotenv

from datetime import datetime

import os

import json

import shutil

from typing import Annotated, TypedDict, List, Dict, Any, Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from pydantic import BaseModel, Field

from pdf2image import convert_from_path

from pytesseract import image_to_string

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from openinference.instrumentation.langchain import LangChainInstrumentor


load_dotenv()
client_qd = QdrantClient("http://localhost:6333") # "http://localhost:6333" | "http://84.252.132.102:6333"
app = FastAPI()
user_id = "1"
LangChainInstrumentor().instrument()

tracer_provider = register(
  project_name="party",
  endpoint="http://localhost:6006/v1/traces",
  auto_instrument=True
)

model = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

query_embedder = OllamaEmbeddings(
    model="hf.co/dengcao/Qwen3-Embedding-0.6B-GGUF:Q8_0",
    base_url="http://localhost:11434",
)

vs_article_storage = QdrantVectorStore.from_existing_collection(embedding=query_embedder, collection_name="articles_collection", url="http://localhost:6333",)
vs_model_markdowns = QdrantVectorStore.from_existing_collection(embedding=query_embedder, collection_name="model_markdowns", url="http://localhost:6333",)

class State(MessagesState):
    recall_memories: List[str]
    theme_of_the_request: str
    final_result = bool

class response(BaseModel):
    reasoner: str = Field(..., description="Provide reasons why this request should or should not be fulfilled based on Russia's laws.")
    fulfillment: str = Field(..., description="Provide ways how to fulfill civilian request by the head of the party if it can be done based on the reasoner otherwise" \
    "write that it is not possible to fulfill.")
    final_result: bool = Field(..., description="Write 'True' if the head of the party can fulfill the request otherwise write 'False'.")

prompt = ChatPromptTemplate([
    ("system",
    """
    You are a useful assistant in the Socialist Party. Your task is to evaluate the possibilities of solving the problems described in the citizens' appeals based on the legislation of the Russian Federation. You will have to determine whether the head of the party can solve the problem described in the appeal or not. 
    You have access to those tools: 
    1. search_memory - This tool is needed in order to find additional information in the RAG system. You can use this information to make your final answer. If you have similar information that you gained frim the RAG you can make the same answer based on this similar information.
    2. tavily_search - This tool is needed to search for information on the Internet. Use it whenever you want or you don't have enough information to form your conclusion. Always check the information several times before forming your conclusion and try to use only verified data. If any articles of the law of the Russian Federation are mentioned, then try to study the article to accurately convey the essence of the article.
    
    Also you will get similar info from the RAG system at the start of your work. It wil be marked like this: <recall_memory> some info </recall_memory>. Those are examples with reasonings on similar problems.
    Please provide your conclusion in Russian.

    Recall_memories: {recall_memories}
    """),
    ("human",
    """
    {messages}
    """)
])


@tool
def search_memory(query: str) -> List[str]:
    """This tool allows you to gain additional information from the RAG that you can use to make decisions"""

    qdrant_filter = Filter(must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))])
    documents = vs_model_markdowns.similarity_search(query, k=3, filter=qdrant_filter)
    
    return [document.page_content for document in documents]


def save_final_result(state: State) -> str:
    data = str({"input": state["theme_of_the_request"][-1], "output": state["messages"][-1].content})

    document = Document(page_content=data, metadata={"user_id": user_id, "timestamp": datetime.now()})
    vs_model_markdowns.add_documents([document])

    return {"final_result": True}


def load_memories(state: State) -> State:
    theme = model.invoke(f"Highlight the main idea of the inputted text in a few sentences in russian. Do not announce anything just highlight the main theme provide only main theme of the full text. Text of the request: {state['messages'][-1].content}")
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=500, chunk_overlap=0)
    theme = text_splitter.split_text(theme.content)

    sim_search = vs_article_storage.similarity_search(theme[0], k=3)

    return {"recall_memories": [document.page_content for document in sim_search], "theme_of_the_request": theme}


def agent(state):
    
    agent = create_agent(
        model=model,
        tools=[search_memory, TavilySearch(max_results=3)],
        response_format=response,
    )

    bound = prompt | agent
    recall_str = ("<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>")
    prediction = bound.invoke({"messages": state["messages"], "recall_memories": recall_str})
    
    response_data = prediction["structured_response"]
    
    return {"messages": [{"role": "assistant", "content": response_data.model_dump_json()}]}


builder = StateGraph(State)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_node(save_final_result)

builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_edge("agent", "save_final_result")
builder.add_edge("save_final_result", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"user_id": "1", "thread_id": "1"}}

client = Client()

dataset = client.datasets.get_dataset(dataset='agent-qa')

def task(input):
    result = graph.invoke({"messages": [(input["question"])]}, config)
    result = result["messages"][-1].content

    return {"answer": result}


def semantic_sim(input, output, expected):

    class Scoring_template(BaseModel):
        semantic_score: int = Field(..., description="Provide score for a semantic equivalence from 1 to 5. " \
        "5 - Perfect semantic match, same meaning." \
        " 4 - Mostly equivalent, minor differences." \
        " 3 - Similar but missing/ adding some information." \
        " 2 - Partially related, significant differences." \
        " 1 - Completely different meaning")
        information: str = Field(..., description="Provide explanation for the scores in a few sentences")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        Your task is to judge other llms answers based on the Ground Truth. You have access to other llms inputs, outputs and expected answers based on questions.
        Evaluate semantic equivalance based on inputs, outputs and ground truth.
        Provide your answer in Russian language.
        """),

        ("human", 
        """
        Input for the llm: {input}.
        Output of the llm: {output}.
        Expected answer (Ground Truth): {expected}.
        """),
    ])

    agent = create_agent(
        model=model,
        tools=[TavilySearch(max_results=3)],
        response_format=Scoring_template,
    )

    bound = prompt | agent
    prediction = bound.invoke({"input": input["question"], "output": output, "expected": expected})
    
    response_data = prediction["structured_response"]
    
    return response_data.model_dump_json()


def information_cover(input, output):
    
    class Scoring_template(BaseModel):
        info_cover: int = Field(..., description="Based on input and output analize how well the answer covers all key information from the input from 1 to 10")
        information: str = Field(..., description="Provide your reasoning here. You can describe your info_cover score in a few sentencies")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        Your goal is to evaluate information coverage based on the input and output data.
        Key aspects to check:
        1. Main points coverage.
        2. Important details. If answer has informmation based on Russian law it has more value.
        3. Critical facts.
        4. Necessary explanations.
        Provide your answer in Russian language.
        """),

        ("human",
        """
        Input for the llm: {input}.
        Output of the llm: {output}.
        """),
    ])

    agent = create_agent(
        model=model,
        tools=[TavilySearch(max_results=3)],
        response_format=Scoring_template,
    )

    bound = prompt | agent
    prediction = bound.invoke({"input": input, "output": output})
    
    response_data = prediction["structured_response"]
    
    return response_data.model_dump_json()


def accuracy(output, expected):
    
    class Scoring_template(BaseModel):
        accuracy: int = Field(..., description="Based on output of the llm and expected output" \
        " evaluate factual accuracy compared to expected output from 1 to 5 based on completeness of the information in the output and precision of details")
        information: str = Field(..., description="Provide your reasoning here. You can describe your accuracy score in a few sentencies")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        Your goal is to evaluate accuracy of the output based on output of the llm and the golden answer. 
        Provide your answer in Russian language.
        """),

        ("human",
        """
        Output of the llm: {output}.
        Golden answer: {expected}
        """),
    ])

    agent = create_agent(
        model=model,
        tools=[TavilySearch(max_results=3)],
        response_format=Scoring_template,
    )

    bound = prompt | agent
    prediction = bound.invoke({"expected": expected, "output": output})
    
    response_data = prediction["structured_response"]
    
    return response_data.model_dump_json()


experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=task,
    evaluators=[semantic_sim, information_cover, accuracy]
)

def task(input):
    result = graph.invoke({"messages": [(input["attributes.input.value"])]}, config)
    result = result["messages"][-1].content

    return {"answer": result}


dataset = client.datasets.get_dataset(dataset='tools_data')

def tools_check(input, output,):

    class Scoring_template(BaseModel):
        tools_invoke_score: int = Field(..., description="Evaluate the justification and appropriateness of tool calls made by the AI assistant. " \
        "Analyze the relationship between the user's input and the actual tools invoked. from 1 to 5. If current tool input has the same information as previous tools inputs give this invoke a slight less score" \
        "5 - Excellent: Tool call is perfectly justified and directly addresses the users request" \
        "4 - Good: Tool call is well-justified with minor room for improvement" \
        "3 - Acceptable: Tool call is somewhat justified but could be more precise" \
        "2 - Poor: Tool call is weakly justified or partially irrelevant" \
        "1 - Unjustified: Tool call is completely unnecessary or inappropriate")
        information: str = Field(..., description="Provide explanation for the scores in a few sentences")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        You are an AI Quality Assurance Specialist evaluating the performance of LangGraph tools. 
        Your task is to assess whether the AI assistant correctly invoked the appropriate tools.
        Here is information about all tools that can be used in the graph:
        1. search_memory - This tool is needed in order to find additional information in the RAG system. Agent can use this information to final answer.
        2. tavily_search - This tool allows graph to search information on the internet.
        Provide your answer in Russian language.
        """),

        ("human",
        """
        Input for the tool: {input}.
        Output of the tool: {output}.
        """),
    ])

    agent = create_agent(
        model=model,
        tools=[TavilySearch(max_results=3)],
        response_format=Scoring_template,
    )

    bound = prompt | agent
    prediction = bound.invoke({"input": input["attributes.input.value"], "output": output})
    
    response_data = prediction["structured_response"]
    
    return response_data.model_dump_json()


experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=task,
    evaluators=[tools_check],
)
