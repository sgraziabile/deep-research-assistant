from datetime import datetime
from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END

from init_llm import get_llm
from state_scope import AgentState, AgentInputState, ClarifyWithUser, ResearchQuestion
from scope_prompts import CLARIFY_PROMPT, RESEARCH_BRIEF_PROMPT

# Load environment variables from .env.local
load_dotenv(".env.local")

# ==== UTILITY FUNCTIONS ====
def get_today_str() -> str: 
    """Returns today's date as a string in the format DD-MM-YYYY."""
    return datetime.now().strftime("%d-%m-%Y")

# ==== CONFIGURATION ====
model = get_llm()

# ==== NODES & GRAPH ====
def research_agent_scope(state: AgentState) -> dict:
    """
    Scope node that evaluates whether to clarify with the user or generate the research brief.
    """
    messages_str = get_buffer_string(state.get("messages", []))
    today_date = get_today_str()
    
    # 1. Check if user needs clarification
    clarify_llm = model.with_structured_output(ClarifyWithUser)
    clarify_res = clarify_llm.invoke(
        CLARIFY_PROMPT.format(today=today_date, messages=messages_str)
    )
    
    if clarify_res.need_clarification:
        return {
            "messages": [AIMessage(content=clarify_res.question)]
        }
    
    # 2. Generate Research Brief if no clarification needed
    brief_llm = model.with_structured_output(ResearchQuestion)
    brief_res = brief_llm.invoke(
        RESEARCH_BRIEF_PROMPT.format(today=today_date, messages=messages_str)
    )
    
    return {
        "research_brief": brief_res.research_brief,
        "messages": [AIMessage(content=clarify_res.verification)]
    }

# ==== GRAPH BUILDER ====
builder = StateGraph(AgentState, input=AgentInputState)
builder.add_node("scope", research_agent_scope)
builder.add_edge(START, "scope")
builder.add_edge("scope", END)

graph = builder.compile()