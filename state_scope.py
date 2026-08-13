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
class ClarifyWithUser(BaseModel):
    """Schema for clarifying questions to ask the user."""

    need_clarification: bool = Field(
        description="Wether the user needs to be asked a clarifying question."
    )
    question: str = Field(
        description="The clarifying question to ask the user."
    )
    verification: str = Field(
        description="Verification message that we will start research after the user has provided the neccesary information."
    )

class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research process."
    )