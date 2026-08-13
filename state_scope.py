import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== STATE DEFINITIONS =====
class AgentInputState(MessagesState):
    """Input state for the agent, only contains messages from the user input."""

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    """

    #Research brief generated from user conversation history
    research_brief: Optional[str]
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[Sequence[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[Sequence[str], operator.add] = []
    # Final formatted research report
    final_report: str

# ==== STRUCTURED OUTPUT SCHEMAS ====