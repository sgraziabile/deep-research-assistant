from typing import TypedDict, Annotated, Sequence
import operator
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv(".env.local")

# --- Types & State ---

# Example of Pydantic for validation
class ResearchBrief(BaseModel):
    topic: str = Field(description="The main topic of the research")
    clarifications: list[str] = Field(default_factory=list, description="Any user clarifications")
    guidelines: list[str] = Field(default_factory=list, description="Guidelines for the research")

# Example of TypedDict, Annotated, Sequence for the Graph State
class AgentState(TypedDict):
    # Annotated with operator.add means these sequences will be appended to rather than overwritten
    messages: Annotated[Sequence[str], operator.add]
    
    # Validation model
    brief: ResearchBrief
    
    # State for research
    research_context: Annotated[Sequence[str], operator.add]
    
    # Final output
    report: str

# --- Nodes ---

def research_agent_scope(state: AgentState):
    """
    Stage 1. Scope: user clarification + brief generation.
    """
    print("--- SCOPE STAGE ---")
    # TODO: Implement user clarification and brief generation logic here
    
    # Return updates to the state
    return {"messages": ["Scope defined (placeholder)"]}

def research_gather_context(state: AgentState):
    """
    Stage 2. Research: gather context from the research.
    """
    print("--- RESEARCH STAGE ---")
    # TODO: Implement research logic here (e.g., using tools like Tavily or web scrapers)
    
    # Return updates to the state
    return {"research_context": ["Context gathered from web/docs (placeholder)"]}

def write_report(state: AgentState):
    """
    Stage 3. Writing: write the report.
    """
    print("--- WRITING STAGE ---")
    # TODO: Implement writing logic here based on brief and research_context
    
    # Return updates to the state
    return {"report": "This is the final research report (placeholder)."}


# --- Graph Setup ---

def build_graph():
    """
    Builds and compiles the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # Add nodes corresponding to the architecture stages
    workflow.add_node("scope", research_agent_scope)
    workflow.add_node("research", research_gather_context)
    workflow.add_node("write", write_report)

    # Define the flow edges
    workflow.add_edge(START, "scope")
    workflow.add_edge("scope", "research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", END)

    # Compile the graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    app = build_graph()
    print("LangGraph project initialized and compiled successfully.")
    
    # To run the graph, you would do something like:
    # initial_state = {
    #     "messages": ["User request for a research topic"],
    #     "brief": ResearchBrief(topic="Default Topic"),
    #     "research_context": [],
    #     "report": ""
    # }
    # result = app.invoke(initial_state)
    # print(result["report"])
