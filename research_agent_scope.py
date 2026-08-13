from typing import Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from dotenv import load_dotenv
from datetime import datetime
from init_llm import get_llm

from state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState
from scope_prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt

# Load environment variables from .env.local
load_dotenv(".env.local")

# ==== UTILITY FUNCTIONS ====
def get_today_str() -> str: 
    """Returns today's date as a string in the format DD-MM-YYYY."""
    return datetime.now().strftime("%d-%m-%Y")

# ==== CONFIGURATION ====

#Initialize model
model = init_chat_model(get_llm(), model_provider="google_genai", temperature=0.0)

# ==== WORKFLOW NODES ====
def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief","__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """

    structured_output_model = model.with_structured_output(ClarifyWithUser)

    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])


    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    
    structured_output_model = model.with_structured_output(ResearchQuestion)

    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"Research brief: {response.research_brief}")]
    }

# ==== STATE GRAPH ====

deep_research_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add nodes to the state graph
deep_research_builder.add_node("clarify_with_user", clarify_with_user)
deep_research_builder.add_node("write_research_brief", write_research_brief)

# Add edges to the state graph
deep_research_builder.add_edge(START, "clarify_with_user")
deep_research_builder.add_edge("write_research_brief", END)

# Compile the workflow 
graph = deep_research_builder.compile()
