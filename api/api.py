from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START, MessagesState
from dotenv import load_dotenv
import os
import json
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from typing import List
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pdf2image import convert_from_path
from pytesseract import image_to_string
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil


load_dotenv()
client_qd = QdrantClient("http://84.252.132.102:6333")
app = FastAPI()
user_id = "1"

model = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


class State(MessagesState):
    recall_memories: List[str]


class Template(BaseModel):
    reasoner: str = Field(..., description="Provide reasons why this request should or should not be fulfilled based on Russia's laws")
    fulfillment: str = Field(..., description="Provide reasons for possibility of fulfilling this reqeust by the head of the party" \
                              "if it is not that hard to fuilfill otherwise write that it is not possible to fulfill")
    final_result: bool = Field(..., description="Write 'True' only if the reqeust if possible to be fulfilled otherwise write 'False'")


parser = PydanticOutputParser(pydantic_object=Template)
format_instructions = parser.get_format_instructions()
fixed_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n"
            "You are working in a russian socialist party and your job is to verify citizen's requests before fulfillment based on laws.\n\n"
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_core_memory, save_recall_memory)"
            " to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored"
            " memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Update your mental model of the user with each new piece of"
            " information.\n"
            "5. Cross-reference new information with existing memories for"
            " consistency.\n"
            "6. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the"
            " user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and"
            " analogies.\n"
            "10. Recall past challenges or successes to inform current"
            " problem-solving.\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n"
            " Please provide your response in the following JSON format:\n{format_instructions}."
        ),
        ("placeholder", "{messages}"),
    ]
)

QUERY_URL = "http://84.252.132.102"
common = {"task": "feature-extraction", "model_kwargs": {"normalize": True}}
query_embedder = HuggingFaceEndpointEmbeddings(model=QUERY_URL, **common)

vs_article_storage = QdrantVectorStore.from_existing_collection(embedding=query_embedder, collection_name="articles_collection", url="http://84.252.132.102:6333",)
vs_model_markdowns = QdrantVectorStore.from_existing_collection(embedding=query_embedder, collection_name="model_markdowns", url="http://84.252.132.102:6333",)


@tool
def search_recall_memories(query: str) -> List[str]:
    """Search for relevant memories."""
    qdrant_filter = Filter(must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))])
    documents = vs_model_markdowns.similarity_search(query, k=3, filter=qdrant_filter)
    
    return [document.page_content for document in documents]


@tool
def save_recall_memory(memory: str) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    document = Document(page_content=memory, metadata={"user_id": user_id})
    vs_model_markdowns.add_documents([document])

    return "Info was written"


tools = [save_recall_memory, search_recall_memories, TavilySearch(max_results=1)]


def agent(state: State) -> State:
    bound = prompt | create_react_agent(model, tools)
    recall_str = ("<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>")
    prediction = bound.invoke({"messages": state["messages"], "recall_memories": recall_str, "format_instructions": format_instructions})
    prediction = parser.invoke(prediction["messages"][-1])

    return {"messages": [{"role": "assistant", "content": prediction.model_dump_json()}], "retry_count": 0}


def load_memories(state: State) -> State:
    theme = model.invoke(f"Highlight the main idea of the inputted text in a few sentences in russian: {state['messages'][-1].content}")
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=500, chunk_overlap=0)
    theme = text_splitter.split_text(theme.content)

    sim_search = vs_article_storage.similarity_search(theme[0], k=3)

    return {"recall_memories": [document.page_content for document in sim_search]}


builder = StateGraph(State)
builder.add_node(load_memories)
builder.add_node(agent)

builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_edge("agent", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"user_id": "1", "thread_id": "1"}}


@app.post("/check_request")
async def check_request(file: UploadFile=File(...)):
    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    images = convert_from_path(file_location)

    text = ""
    for image in images:
        page_text = image_to_string(image, lang="rus+eng")
        text += f"{page_text}"
    
    preds = graph.invoke({"messages": [(text)]}, config)

    os.remove(file_location)

    return {"Answer": json.loads(preds["messages"][-1].content)}
