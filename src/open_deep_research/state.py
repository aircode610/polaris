"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class MCTSNode(BaseModel):
    """Represents a node in the MCTS search tree for research planning."""
    
    node_id: str = Field(description="Unique identifier for this node")
    parent_id: Optional[str] = Field(default=None, description="Parent node ID")
    children_ids: list[str] = Field(default_factory=list, description="Child node IDs")
    depth: int = Field(description="Depth in the tree (0 for root)")
    
    # Research path information
    research_angle: str = Field(
        description="The specific research angle/perspective this node represents"
    )
    research_focus: str = Field(
        description="Detailed description of what to focus on in this research path"
    )
    
    # MCTS statistics
    visit_count: int = Field(default=0, description="Number of times this node was visited")
    total_value: float = Field(default=0.0, description="Cumulative value from simulations")
    
    # Evaluation scores (0-1 scale)
    comprehensiveness_score: float = Field(
        default=0.0,
        description="Expected comprehensiveness of this research path (0-1)",
        ge=0.0,
        le=1.0
    )
    insight_score: float = Field(
        default=0.0,
        description="Expected depth and insight potential (0-1)",
        ge=0.0,
        le=1.0
    )
    instruction_following_score: float = Field(
        default=0.0,
        description="Alignment with original research brief (0-1)",
        ge=0.0,
        le=1.0
    )
    feasibility_score: float = Field(
        default=0.0,
        description="Likelihood of finding good information (0-1)",
        ge=0.0,
        le=1.0
    )
    
    # State flags
    is_terminal: bool = Field(default=False, description="Whether this is a terminal node")
    is_fully_expanded: bool = Field(default=False, description="All children have been generated")


class ResearchAngle(BaseModel):
    """A single research angle for node expansion."""
    
    research_angle: str = Field(
        description="Name of the research angle (4-8 words)"
    )
    research_focus: str = Field(
        description="Detailed description of the research focus (2-4 sentences)"
    )


class ResearchAngles(BaseModel):
    """Collection of research angles for node expansion."""
    
    angles: list[ResearchAngle] = Field(
        description="List of diverse research angles. Each angle has a 'research_angle' (name, 4-8 words) and 'research_focus' (detailed description, 2-4 sentences)",
        min_items=1,
        max_items=5
    )


class PathEvaluation(BaseModel):
    """Evaluation of a research path without executing research."""
    
    comprehensiveness_score: float = Field(
        description="Expected comprehensiveness score (0-1): Would this path cover important aspects?",
        ge=0.0,
        le=1.0
    )
    insight_score: float = Field(
        description="Expected insight/depth score (0-1): Would it uncover deep insights?",
        ge=0.0,
        le=1.0
    )
    instruction_following_score: float = Field(
        description="Alignment with research brief (0-1): How well does it match objectives?",
        ge=0.0,
        le=1.0
    )
    feasibility_score: float = Field(
        description="Likelihood of finding good information (0-1): Is information available?",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of the scores (2-3 sentences)"
    )


class ResearchStrategy(BaseModel):
    """Structured research strategy to guide the supervisor."""
    
    priority_angles: list[str] = Field(
        description="List of research angles ranked by priority",
        default_factory=list
    )
    recommended_focus_areas: list[str] = Field(
        description="Key areas to focus research efforts",
        default_factory=list
    )
    suggested_methodologies: list[str] = Field(
        description="Recommended research methodologies",
        default_factory=list
    )
    exploration_summary: str = Field(
        description="Summary of MCTS exploration process and best path found",
        default=""
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    research_strategy: Optional[ResearchStrategy] = None

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    research_strategy: Optional[ResearchStrategy]

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


class MCTSPlannerState(TypedDict):
    """State for the MCTS planning phase."""
    
    research_brief: str
    nodes: dict[str, MCTSNode]
    root_node_id: str
    current_iteration: int
    max_iterations: int
    max_depth: int
    best_path: list[str]
    research_strategy: Optional[ResearchStrategy]