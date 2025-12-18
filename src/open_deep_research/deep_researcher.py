"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import math
import uuid
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    mcts_evaluation_prompt,
    mcts_expansion_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    MCTSNode,
    MCTSPlannerState,
    PathEvaluation,
    ResearchAngles,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    ResearchStrategy,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["mcts_planner"]]:
    """Transform user messages into a structured research brief and proceed to MCTS planning.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the MCTS planner and research supervisor.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to MCTS planner with research brief
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    return Command(
        goto="mcts_planner", 
        update={
            "research_brief": response.research_brief,
        }
    )


async def mcts_planner(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """MCTS planning phase to explore research paths and create optimal strategy.
    
    This function performs a lightweight exploration of different research directions
    using Monte Carlo Tree Search, then provides structured guidance to the supervisor.
    
    Args:
        state: Current agent state with research brief
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with strategy
    """
    
    # Step 1: Get configuration and set MCTS parameters
    configurable = Configuration.from_runnable_config(config)
    research_brief = state.get("research_brief", "")
    
    # MCTS configuration (using defaults if not in config)
    max_iterations = getattr(configurable, 'mcts_max_iterations', 10)
    max_depth = getattr(configurable, 'mcts_max_depth', 3)
    branching_factor = getattr(configurable, 'mcts_branching_factor', 3)
    exploration_constant = getattr(configurable, 'mcts_exploration_constant', 1.414)
    
    # Step 2: Initialize MCTS tree with root node
    root_node = MCTSNode(
        node_id="root",
        parent_id=None,
        depth=0,
        research_angle="Root: Original Research Brief",
        research_focus=research_brief
    )
    
    mcts_state: MCTSPlannerState = {
        "research_brief": research_brief,
        "nodes": {"root": root_node},
        "root_node_id": "root",
        "current_iteration": 0,
        "max_iterations": max_iterations,
        "max_depth": max_depth,
        "best_path": ["root"],
        "research_strategy": None
    }
    
    # Step 3: Run MCTS iterations
    for iteration in range(max_iterations):
        mcts_state["current_iteration"] = iteration
        
        # Selection: Select node using UCB1
        selected_node_id = _select_node(mcts_state, exploration_constant)
        selected_node = mcts_state["nodes"][selected_node_id]
        
        # Expansion: Generate child nodes if not terminal
        if not selected_node.is_terminal and not selected_node.is_fully_expanded:
            new_node_ids = await _expand_node(
                selected_node_id, 
                mcts_state, 
                branching_factor,
                configurable,
                config
            )
        else:
            new_node_ids = [selected_node_id]
        
        # Simulation: Evaluate new nodes
        for node_id in new_node_ids:
            value = await _evaluate_node(node_id, mcts_state, configurable, config)
            
            # Backpropagation: Update values up the tree
            _backpropagate(node_id, value, mcts_state)
    
    # Step 4: Create research strategy from best path
    research_strategy = _create_research_strategy(mcts_state)
    
    # Step 5: Enhance supervisor prompt with strategy
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    # Add strategy context to supervisor prompt if available
    if research_strategy and research_strategy.priority_angles:
        strategy_context = f"\n\n<Research Strategy from MCTS Planning>\n"
        strategy_context += "The MCTS planner has explored multiple research paths and identified the following promising directions. "
        strategy_context += "**Use these as prioritized starting points, but do not limit yourself to only these angles.** "
        strategy_context += "You should also explore other relevant topics that may be important for comprehensive research.\n\n"
        strategy_context += "**Priority Research Angles (suggested starting points, in order of importance):**\n"
        for i, angle in enumerate(research_strategy.priority_angles[:5], 1):
            strategy_context += f"  {i}. {angle}\n"
        strategy_context += "\n**Recommended Focus Areas:**\n"
        for i, area in enumerate(research_strategy.recommended_focus_areas[:5], 1):
            strategy_context += f"  {i}. {area}\n"
        strategy_context += f"\n**Planning Summary:** {research_strategy.exploration_summary}\n"
        strategy_context += "\n**Guidance:** Start your research with the above priority angles, but also identify and explore other important aspects "
        strategy_context += "of the research question that may not be in this list. The MCTS plan is a helpful guide, not a strict constraint. "
        strategy_context += "Your goal is comprehensive research coverage, so balance following the plan with exploring other relevant topics.\n"
        strategy_context += "</Research Strategy from MCTS Planning>"
        supervisor_system_prompt += strategy_context
    elif research_strategy:
        # Even if no priority angles, still include the strategy context
        strategy_context = f"\n\n<Research Strategy from MCTS Planning>\n"
        strategy_context += f"**Planning Summary:** {research_strategy.exploration_summary}\n"
        strategy_context += "Use this as general guidance, but ensure you explore all relevant aspects of the research question.\n"
        strategy_context += "</Research Strategy from MCTS Planning>"
        supervisor_system_prompt += strategy_context
    
    return Command(
        goto="research_supervisor",
        update={
            "research_strategy": research_strategy,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=research_brief)
                ]
            }
        }
    )


def _select_node(mcts_state: MCTSPlannerState, exploration_constant: float) -> str:
    """Select the most promising node for expansion using UCB1."""
    current_node_id = mcts_state["root_node_id"]
    nodes = mcts_state["nodes"]
    
    while True:
        current_node = nodes[current_node_id]
        
        # Stop if terminal or needs expansion
        if current_node.is_terminal or not current_node.is_fully_expanded or not current_node.children_ids:
            return current_node_id
        
        # Select best child using UCB1
        parent_visits = current_node.visit_count
        best_child_id = max(
            current_node.children_ids,
            key=lambda child_id: _ucb1_score(
                nodes[child_id],
                parent_visits,
                exploration_constant
            )
        )
        
        current_node_id = best_child_id
        
        # Prevent infinite loops at max depth
        if nodes[current_node_id].depth >= mcts_state["max_depth"]:
            nodes[current_node_id].is_terminal = True
            return current_node_id


def _ucb1_score(node: MCTSNode, parent_visits: int, exploration_constant: float) -> float:
    """Calculate UCB1 score for node selection."""
    if node.visit_count == 0:
        return float('inf')
    
    exploitation = node.total_value / node.visit_count if node.visit_count > 0 else 0.0
    exploration = exploration_constant * math.sqrt(
        math.log(parent_visits) / node.visit_count
    )
    return exploitation + exploration


async def _expand_node(
    node_id: str,
    mcts_state: MCTSPlannerState,
    branching_factor: int,
    configurable: Configuration,
    config: RunnableConfig
) -> list[str]:
    """Expand a node by generating child nodes representing different research angles."""
    
    parent_node = mcts_state["nodes"][node_id]
    
    # Don't expand beyond max depth
    if parent_node.depth >= mcts_state["max_depth"]:
        parent_node.is_terminal = True
        parent_node.is_fully_expanded = True
        return []
    
    # Generate diverse research angles using LLM
    expansion_prompt_text = mcts_expansion_prompt.format(
        research_brief=mcts_state["research_brief"],
        current_angle=parent_node.research_angle,
        current_focus=parent_node.research_focus,
        depth=parent_node.depth,
        max_depth=mcts_state["max_depth"],
        branching_factor=branching_factor,
        date=get_today_str()
    )
    
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    research_model = (
        configurable_model
        .with_structured_output(ResearchAngles)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    try:
        response = await research_model.ainvoke([
            HumanMessage(content=expansion_prompt_text)
        ])
    except Exception as e:
        parent_node.is_fully_expanded = True
        parent_node.is_terminal = True
        return []
    
    # Ensure we have a valid response with angles
    if not hasattr(response, 'angles') or not response.angles:
        parent_node.is_fully_expanded = True
        parent_node.is_terminal = True
        return []
    
    # Create child nodes from suggested angles
    new_node_ids = []
    for idx, angle in enumerate(response.angles[:branching_factor]):
        child_id = f"{node_id}_c{idx}_{uuid.uuid4().hex[:6]}"
        
        # Handle both Pydantic model and dict formats
        if isinstance(angle, dict):
            research_angle = angle.get("research_angle", "")
            research_focus = angle.get("research_focus", "")
        else:
            # Pydantic model - use attribute access
            research_angle = angle.research_angle
            research_focus = angle.research_focus
        
        child_node = MCTSNode(
            node_id=child_id,
            parent_id=node_id,
            depth=parent_node.depth + 1,
            research_angle=research_angle,
            research_focus=research_focus
        )
        
        mcts_state["nodes"][child_id] = child_node
        new_node_ids.append(child_id)
    
    # Update parent
    parent_node.children_ids.extend(new_node_ids)
    parent_node.is_fully_expanded = True
    
    return new_node_ids


async def _evaluate_node(
    node_id: str,
    mcts_state: MCTSPlannerState,
    configurable: Configuration,
    config: RunnableConfig
) -> float:
    """Evaluate a research path without executing full research."""
    node = mcts_state["nodes"][node_id]
    
    # Create evaluation prompt
    eval_prompt_text = mcts_evaluation_prompt.format(
        research_brief=mcts_state["research_brief"],
        current_angle=node.research_angle,
        current_focus=node.research_focus,
        depth=node.depth,
        date=get_today_str()
    )
    
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    evaluation_model = (
        configurable_model
        .with_structured_output(PathEvaluation)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    try:
        evaluation = await evaluation_model.ainvoke([
            HumanMessage(content=eval_prompt_text)
        ])
    except Exception as e:
        # Return conservative estimate on failure
        return 0.5
    
    # Update node scores
    node.comprehensiveness_score = evaluation.comprehensiveness_score
    node.insight_score = evaluation.insight_score
    node.instruction_following_score = evaluation.instruction_following_score
    node.feasibility_score = evaluation.feasibility_score
    
    # Return combined score as value (weighted average)
    combined_score = (
        0.30 * node.comprehensiveness_score +
        0.30 * node.insight_score +
        0.25 * node.instruction_following_score +
        0.15 * node.feasibility_score
    )
    return combined_score


def _backpropagate(node_id: str, value: float, mcts_state: MCTSPlannerState):
    """Backpropagate the evaluation value up the tree."""
    current_node_id = node_id
    nodes = mcts_state["nodes"]
    
    while current_node_id is not None:
        node = nodes[current_node_id]
        node.visit_count += 1
        node.total_value += value
        current_node_id = node.parent_id


def _create_research_strategy(mcts_state: MCTSPlannerState) -> ResearchStrategy:
    """Extract the best research path and create a structured strategy."""
    nodes = mcts_state["nodes"]
    root_id = mcts_state["root_node_id"]
    
    # Find best path from root using average values
    best_path = [root_id]
    current_id = root_id
    
    while nodes[current_id].children_ids:
        children_with_visits = [
            cid for cid in nodes[current_id].children_ids 
            if nodes[cid].visit_count > 0
        ]
        
        if not children_with_visits:
            break
        
        best_child_id = max(
            children_with_visits,
            key=lambda cid: nodes[cid].total_value / nodes[cid].visit_count if nodes[cid].visit_count > 0 else 0.0
        )
        best_path.append(best_child_id)
        current_id = best_child_id
    
    mcts_state["best_path"] = best_path
    
    # Build strategy from best path
    priority_angles = []
    focus_areas = []
    methodologies = []
    
    # Add nodes from best path
    for node_id in best_path[1:]:  # Skip root
        node = nodes[node_id]
        priority_angles.append(node.research_angle)
        focus_areas.append(node.research_focus)
    
    # Create exploration summary
    total_nodes = len(nodes)
    visited_nodes = [n for n in nodes.values() if n.visit_count > 0]
    max_value = max(
        (n.total_value / n.visit_count for n in visited_nodes),
        default=0.0
    )
    
    exploration_summary = (
        f"Explored {total_nodes} research paths across {mcts_state['max_depth']} depth levels. "
        f"Best path achieves expected quality score of {max_value:.2f}. "
        f"Strategy based on {mcts_state['max_iterations']} MCTS iterations."
    )
    
    return ResearchStrategy(
        priority_angles=priority_angles,
        recommended_focus_areas=focus_areas,
        suggested_methodologies=methodologies,
        exploration_summary=exploration_summary
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
                "research_strategy": state.get("research_strategy")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", ""),
                        "research_strategy": state.get("research_strategy")
                    }
                )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # Return successful report generation
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("mcts_planner", mcts_planner)                    # MCTS planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("write_research_brief", "mcts_planner")           # Brief to MCTS planning
deep_researcher_builder.add_edge("mcts_planner", "research_supervisor")           # MCTS planning to supervisor
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()