# MCTS Feature in Deep Research Agent - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [MCTS Architecture](#mcts-architecture)
3. [State Management](#state-management)
4. [MCTS Algorithm Components](#mcts-algorithm-components)
5. [Integration with Research Pipeline](#integration-with-research-pipeline)
6. [Configuration Parameters](#configuration-parameters)
7. [Flow Diagram](#flow-diagram)
8. [Key Implementation Details](#key-implementation-details)

---

## Overview

The **Monte Carlo Tree Search (MCTS)** feature is a planning phase that runs **before** actual research execution. It explores different research paths to identify the most promising angles and creates a strategic plan that guides the research supervisor.

### Purpose
- **Explore** multiple research directions without executing expensive research
- **Evaluate** potential research paths on multiple quality dimensions
- **Prioritize** research angles based on expected value
- **Guide** the supervisor with a structured strategy

### Position in Pipeline
```
User Input → Clarification → Research Brief → **MCTS Planner** → Research Supervisor → Final Report
```

---

## MCTS Architecture

### High-Level Flow

```
1. Initialize root node with research brief
2. For each iteration (default: 10):
   a. SELECT: Choose node using UCB1 algorithm
   b. EXPAND: Generate child nodes (research angles) using LLM
   c. EVALUATE: Score each new node using LLM
   d. BACKPROPAGATE: Update statistics up the tree
3. Extract best path and create research strategy
4. Pass strategy to supervisor
```

### Tree Structure

Each node in the MCTS tree represents a **research path**:
- **Root Node**: Original research brief
- **Child Nodes**: Different research angles/perspectives
- **Depth**: Maximum 3 levels by default
- **Branching**: 3 children per node by default

---

## State Management

### MCTSPlannerState

Located in `state.py`, this TypedDict manages the entire MCTS tree:

```python
class MCTSPlannerState(TypedDict):
    research_brief: str                    # Original research question
    nodes: dict[str, MCTSNode]            # All nodes in the tree
    root_node_id: str                      # ID of root node ("root")
    current_iteration: int                 # Current iteration number
    max_iterations: int                    # Total iterations (default: 10)
    max_depth: int                         # Maximum tree depth (default: 3)
    best_path: list[str]                   # Node IDs of best path found
    research_strategy: Optional[ResearchStrategy]  # Final strategy output
```

### MCTSNode

Each node represents a research path with rich metadata:

```python
class MCTSNode(BaseModel):
    # Identity
    node_id: str                           # Unique identifier
    parent_id: Optional[str]               # Parent node ID
    children_ids: list[str]                # Child node IDs
    depth: int                             # Depth in tree (0 = root)
    
    # Research Content
    research_angle: str                    # Name of research angle (4-8 words)
    research_focus: str                    # Detailed description (2-4 sentences)
    
    # MCTS Statistics
    visit_count: int                       # Number of times visited
    total_value: float                     # Cumulative value from evaluations
    
    # Quality Scores (0.0-1.0)
    comprehensiveness_score: float         # Coverage of important aspects
    insight_score: float                   # Depth and analytical potential
    instruction_following_score: float     # Alignment with research brief
    feasibility_score: float              # Likelihood of finding information
    
    # State Flags
    is_terminal: bool                      # Cannot expand further
    is_fully_expanded: bool                # All children generated
```

### Key State Transitions

1. **Initialization** (lines 196-214 in `deep_researcher.py`):
   - Create root node with research brief
   - Initialize empty nodes dictionary
   - Set iteration counters

2. **During Iterations** (lines 217-241):
   - Nodes are added to `nodes` dictionary
   - Statistics updated in-place
   - Best path tracked

3. **Finalization** (lines 243-244):
   - Extract best path from tree
   - Create `ResearchStrategy` object
   - Pass to supervisor

---

## MCTS Algorithm Components

### 1. Selection Phase (`_select_node`)

**Purpose**: Choose the most promising node to explore using UCB1 (Upper Confidence Bound) algorithm.

**Algorithm** (lines 294-323):
```python
def _select_node(mcts_state, exploration_constant):
    current_node_id = root_node_id
    
    while True:
        node = nodes[current_node_id]
        
        # Stop if terminal or needs expansion
        if node.is_terminal or not node.is_fully_expanded:
            return current_node_id
        
        # Select best child using UCB1
        best_child_id = max(
            node.children_ids,
            key=lambda child_id: _ucb1_score(
                nodes[child_id],
                node.visit_count,
                exploration_constant
            )
        )
        
        current_node_id = best_child_id
```

**UCB1 Formula** (lines 325-334):
```python
def _ucb1_score(node, parent_visits, exploration_constant):
    if node.visit_count == 0:
        return float('inf')  # Prioritize unvisited nodes
    
    exploitation = node.total_value / node.visit_count
    exploration = exploration_constant * sqrt(
        log(parent_visits) / node.visit_count
    )
    return exploitation + exploration
```

**Key Points**:
- **Exploitation**: Average value of node (higher is better)
- **Exploration**: Encourages visiting less-explored nodes
- **Balance**: `exploration_constant` (default: 1.414) controls exploration vs exploitation
- **Unvisited nodes**: Get infinite score (highest priority)

### 2. Expansion Phase (`_expand_node`)

**Purpose**: Generate child nodes representing different research angles.

**Process** (lines 337-423):
1. Check if expansion is allowed (not at max depth)
2. Call LLM with `mcts_expansion_prompt` to generate diverse angles
3. Create child nodes for each angle
4. Update parent's `children_ids` and mark as fully expanded

**LLM Prompt** (`mcts_expansion_prompt` in `prompts.py`):
- Takes: research brief, current angle, current focus, depth, branching factor
- Returns: `ResearchAngles` with list of `ResearchAngle` objects
- Each angle has: `research_angle` (name) and `research_focus` (description)

**Node Creation**:
```python
for angle in response.angles[:branching_factor]:
    child_id = f"{node_id}_c{idx}_{uuid.uuid4().hex[:6]}"
    child_node = MCTSNode(
        node_id=child_id,
        parent_id=node_id,
        depth=parent_node.depth + 1,
        research_angle=angle.research_angle,
        research_focus=angle.research_focus
    )
    mcts_state["nodes"][child_id] = child_node
```

**Key Points**:
- Uses LLM to generate **diverse** research angles
- Each child represents a different perspective/approach
- Maximum `branching_factor` children (default: 3)
- Stops at `max_depth` (default: 3)

### 3. Evaluation Phase (`_evaluate_node`)

**Purpose**: Score a research path without executing actual research.

**Process** (lines 426-479):
1. Call LLM with `mcts_evaluation_prompt`
2. LLM evaluates on 4 dimensions (0.0-1.0):
   - **Comprehensiveness**: Coverage of important aspects
   - **Insight**: Depth and analytical potential
   - **Instruction Following**: Alignment with research brief
   - **Feasibility**: Likelihood of finding good information
3. Store scores in node
4. Return combined weighted score

**LLM Prompt** (`mcts_evaluation_prompt` in `prompts.py`):
- Takes: research brief, current angle, current focus, depth
- Returns: `PathEvaluation` with 4 scores + reasoning

**Score Calculation**:
```python
combined_score = (
    0.30 * comprehensiveness_score +
    0.30 * insight_score +
    0.25 * instruction_following_score +
    0.15 * feasibility_score
)
```

**Key Points**:
- **No actual research**: Only predicts quality
- **Multi-dimensional**: 4 different quality metrics
- **Weighted combination**: Different weights for different aspects
- **Fast**: Much faster than executing research

### 4. Backpropagation Phase (`_backpropagate`)

**Purpose**: Update statistics up the tree from evaluated node to root.

**Algorithm** (lines 482-491):
```python
def _backpropagate(node_id, value, mcts_state):
    current_node_id = node_id
    nodes = mcts_state["nodes"]
    
    while current_node_id is not None:
        node = nodes[current_node_id]
        node.visit_count += 1
        node.total_value += value
        current_node_id = node.parent_id
```

**Key Points**:
- **Updates all ancestors**: From evaluated node to root
- **Increments visit_count**: Tracks how often path was explored
- **Accumulates total_value**: Sum of all evaluation scores
- **Enables UCB1**: Provides data for future selection decisions

---

## Integration with Research Pipeline

### Entry Point: `mcts_planner` Function

**Location**: `deep_researcher.py`, lines 172-291

**Input State** (`AgentState`):
- `research_brief`: Research question from previous step

**Output State**:
- `research_strategy`: `ResearchStrategy` object with prioritized angles
- `supervisor_messages`: Enhanced system prompt with MCTS guidance

### ResearchStrategy Object

Created in `_create_research_strategy` (lines 494-551):

```python
class ResearchStrategy(BaseModel):
    priority_angles: list[str]              # Ranked research angles
    recommended_focus_areas: list[str]      # Key focus areas
    suggested_methodologies: list[str]      # Research methods (currently unused)
    exploration_summary: str                # Summary of MCTS exploration
```

**Best Path Extraction**:
1. Start at root node
2. Follow path of highest average value children
3. Extract `research_angle` from each node in path
4. Create priority list

### Supervisor Integration

**Enhanced Prompt** (lines 247-277):
The MCTS strategy is injected into the supervisor's system prompt:

```python
strategy_context = f"""
<Research Strategy from MCTS Planning>
The MCTS planner has explored multiple research paths and identified the following promising directions.

**Priority Research Angles (suggested starting points, in order of importance):**
  1. {angle_1}
  2. {angle_2}
  ...

**Recommended Focus Areas:**
  1. {focus_1}
  2. {focus_2}
  ...

**Planning Summary:** {exploration_summary}

**Guidance:** Start your research with the above priority angles, but also identify and explore other important aspects of the research question that may not be in this list.
</Research Strategy from MCTS Planning>
"""
```

**Key Design Decision**:
- MCTS strategy is **guidance**, not a constraint
- Supervisor can explore beyond MCTS suggestions
- Balances planning with flexibility

---

## Configuration Parameters

### MCTS-Specific Parameters

Located in `configuration.py`, but currently use **defaults** (not exposed in config):

```python
# Default values (lines 191-194 in deep_researcher.py)
max_iterations = getattr(configurable, 'mcts_max_iterations', 10)
max_depth = getattr(configurable, 'mcts_max_depth', 3)
branching_factor = getattr(configurable, 'mcts_branching_factor', 3)
exploration_constant = getattr(configurable, 'mcts_exploration_constant', 1.414)
```

### Parameter Effects

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_iterations` | 10 | More iterations = better exploration, but slower |
| `max_depth` | 3 | Deeper trees = more detailed planning, but more LLM calls |
| `branching_factor` | 3 | More branches = more diversity, but more LLM calls |
| `exploration_constant` | 1.414 | Higher = more exploration, lower = more exploitation |

### Model Configuration

Uses the same model as research:
- `configurable.research_model` (default: "openai:gpt-4.1")
- `configurable.research_model_max_tokens` (default: 10000)
- Structured output with retry logic

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MCTS PLANNER PHASE                        │
└─────────────────────────────────────────────────────────────┘

1. INITIALIZATION
   ┌─────────────────┐
   │ Research Brief  │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Create Root     │
   │  Node            │
   └────────┬────────┘

2. MCTS ITERATIONS (10x)
   ┌─────────────────────────────────────────┐
   │  For each iteration:                    │
   │                                         │
   │  ┌──────────────┐                      │
   │  │   SELECT     │ ← UCB1 algorithm      │
   │  │   Node       │                       │
   │  └──────┬───────┘                       │
   │         │                                │
   │         ▼                                │
   │  ┌──────────────┐                      │
   │  │   EXPAND     │ ← LLM generates      │
   │  │   Node       │   research angles     │
   │  └──────┬───────┘                       │
   │         │                                │
   │         ▼                                │
   │  ┌──────────────┐                      │
   │  │  EVALUATE    │ ← LLM scores path     │
   │  │   Nodes      │   (4 dimensions)       │
   │  └──────┬───────┘                       │
   │         │                                │
   │         ▼                                │
   │  ┌──────────────┐                      │
   │  │ BACKPROPAGATE│ ← Update statistics  │
   │  │   Values      │   up tree            │
   │  └──────────────┘                       │
   └─────────────────────────────────────────┘

3. STRATEGY EXTRACTION
   ┌─────────────────┐
   │  Find Best Path  │ ← Follow highest value
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Create Strategy │ ← ResearchStrategy object
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Enhance Prompt  │ ← Add to supervisor
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Pass to        │
   │  Supervisor     │
   └─────────────────┘
```

---

## Key Implementation Details

### 1. Tree Growth Pattern

**Iteration 1**:
```
root (visited: 1, value: 0.7)
├── child_1 (visited: 1, value: 0.7)
├── child_2 (visited: 1, value: 0.7)
└── child_3 (visited: 1, value: 0.7)
```

**Iteration 2** (UCB1 selects child_1):
```
root (visited: 2, value: 1.4)
├── child_1 (visited: 2, value: 1.5) ← Selected
│   ├── grandchild_1 (visited: 1, value: 0.8)
│   ├── grandchild_2 (visited: 1, value: 0.8)
│   └── grandchild_3 (visited: 1, value: 0.8)
├── child_2 (visited: 1, value: 0.7)
└── child_3 (visited: 1, value: 0.7)
```

### 2. UCB1 Selection Example

Given:
- Parent visits: 10
- Child A: visits=5, avg_value=0.8
- Child B: visits=2, avg_value=0.6
- Exploration constant: 1.414

**Child A UCB1**:
```
exploitation = 0.8
exploration = 1.414 * sqrt(log(10) / 5) = 1.414 * sqrt(0.46) = 0.96
UCB1 = 0.8 + 0.96 = 1.76
```

**Child B UCB1**:
```
exploitation = 0.6
exploration = 1.414 * sqrt(log(10) / 2) = 1.414 * sqrt(1.15) = 1.52
UCB1 = 0.6 + 1.52 = 2.12
```

**Result**: Child B selected (higher UCB1 despite lower value - exploration bonus)

### 3. Evaluation Scoring

**Example Node Evaluation**:
```
Comprehensiveness: 0.85  (30% weight) = 0.255
Insight:           0.70  (30% weight) = 0.210
Instruction:       0.90  (25% weight) = 0.225
Feasibility:       0.75  (15% weight) = 0.113
────────────────────────────────────────────
Combined Score:                          0.803
```

### 4. Best Path Extraction

**Algorithm** (lines 499-517):
```python
best_path = [root_id]
current_id = root_id

while nodes[current_id].children_ids:
    # Find children that were actually visited
    children_with_visits = [
        cid for cid in nodes[current_id].children_ids 
        if nodes[cid].visit_count > 0
    ]
    
    if not children_with_visits:
        break
    
    # Select child with highest average value
    best_child_id = max(
        children_with_visits,
        key=lambda cid: nodes[cid].total_value / nodes[cid].visit_count
    )
    
    best_path.append(best_child_id)
    current_id = best_child_id
```

**Result**: List of node IDs representing the highest-value path from root to leaf

### 5. Error Handling

**Expansion Failures** (lines 383-392):
- If LLM call fails → Mark node as terminal and fully expanded
- If no angles returned → Mark node as terminal
- Prevents infinite loops

**Evaluation Failures** (lines 462-464):
- If LLM call fails → Return conservative score (0.5)
- Ensures tree exploration continues

**Depth Limits** (lines 349-352):
- Nodes at `max_depth` are marked terminal
- Prevents infinite expansion

---

## Summary

### What MCTS Does

1. **Explores** research space before execution
2. **Evaluates** paths on 4 quality dimensions
3. **Prioritizes** angles using UCB1 algorithm
4. **Guides** supervisor with structured strategy

### Key Benefits

- **Efficiency**: Identifies promising paths without expensive research
- **Quality**: Multi-dimensional evaluation ensures comprehensive planning
- **Flexibility**: Supervisor can deviate from plan if needed
- **Scalability**: Configurable parameters for different use cases

### Design Philosophy

- **Planning before execution**: Invest time upfront to save time later
- **LLM-powered exploration**: Uses AI to generate diverse research angles
- **Statistical decision-making**: UCB1 balances exploration and exploitation
- **Non-constraining guidance**: Strategy is helpful but not restrictive

---

## Presentation Tips

1. **Start with the problem**: Why do we need planning before research?
2. **Show the tree structure**: Visual representation helps
3. **Walk through one iteration**: Selection → Expansion → Evaluation → Backpropagation
4. **Explain UCB1**: Balance between exploration and exploitation
5. **Show integration**: How strategy flows to supervisor
6. **Highlight benefits**: Efficiency, quality, flexibility

---

## Code References

- **Main MCTS function**: `deep_researcher.py`, lines 172-291
- **Selection**: `_select_node`, lines 294-323
- **Expansion**: `_expand_node`, lines 337-423
- **Evaluation**: `_evaluate_node`, lines 426-479
- **Backpropagation**: `_backpropagate`, lines 482-491
- **Strategy creation**: `_create_research_strategy`, lines 494-551
- **State definitions**: `state.py`, lines 51-100, 222-232
- **Prompts**: `prompts.py`, lines 372-454
