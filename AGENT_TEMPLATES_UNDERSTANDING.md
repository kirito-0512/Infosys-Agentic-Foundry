# Infosys Agentic Foundry - Agent Templates Understanding

## Executive Summary

The Infosys Agentic Foundry is an enterprise-grade framework for creating, managing, and deploying AI agents. It uses a **template-based architecture** where different agent behavioral patterns (templates) can be instantiated with specific goals, workflows, and tools to create specialized agents.

---

## Core Concepts

### 1. Agent Templates

**Agent templates** are pre-defined behavioral patterns that determine how an agent reasons, plans, and executes tasks. Think of them as "personality types" or "operational modes" for AI agents.

#### Available Templates:

| Template Name | Type | Description | Use Case |
|--------------|------|-------------|----------|
| **React Agent** | Single-step | Uses Reasoning-Action-Observation loop | Simple tasks requiring tool usage with step-by-step logic |
| **React Critic Agent** | Single-step + Review | React pattern with self-validation | Tasks requiring quality assurance at each step |
| **Planner-Executor Agent** | Two-phase | Separates planning from execution | Tasks requiring upfront strategy before action |
| **Multi Agent** | Three-phase | Planner → Executor → Critic | Complex workflows needing planning, execution, and evaluation |
| **Meta Agent** | Orchestrator | Delegates tasks to worker agents | Multi-agent coordination scenarios |
| **Planner-Meta Agent** | Planning Orchestrator | Strategic planning + worker delegation | Advanced multi-agent workflows with planning |
| **Hybrid Agent** | Mixed | Combines multiple patterns | Complex scenarios requiring pattern switching |
| **Simple AI Agent** | Minimal | Basic chat without tools | Conversational agents without tool access |

---

### 2. Agent Creation Flow

#### Step 1: User Submits Agent Configuration

```json
{
  "agent_name": "DataAnalyzer",
  "agent_type": "react_agent",
  "agent_goal": "Analyze customer data and generate insights",
  "workflow_description": "1. Read data files\n2. Process and clean data\n3. Generate statistical insights",
  "model_name": "gpt-4",
  "tools_id": ["tool_csv_reader", "tool_data_processor"],
  "email_id": "user@company.com"
}
```

#### Step 2: API Processing (`/agents/onboard`)

1. **Permission Check**: Validates user has permission to create agents (RBAC)
2. **Template Selection**: Routes to appropriate template service based on `agent_type`
3. **Template Instantiation**: Calls template-specific onboarding logic

#### Step 3: System Prompt Generation

Each template dynamically generates a system prompt using:
- **Agent Name & Goal**: What the agent is and what it should achieve
- **Workflow Description**: Step-by-step process to follow
- **Tool Descriptions**: Capabilities and limitations of available tools
- **LLM Context**: Model-specific instructions

**Example for React Agent** (simplified):
```
You are DataAnalyzer, an AI agent specialized in analyzing customer data.

Your goal: Analyze customer data and generate insights

Workflow:
1. Read data files using the CSV reader tool
2. Process and clean data using the data processor tool
3. Generate statistical insights

Available Tools:
- csv_reader: Reads CSV files. Limitation: Max 10MB file size
- data_processor: Processes data. Limitation: Cannot handle null values

Follow the ReAct pattern: Reason → Act → Observe → Repeat
```

#### Step 4: Database Storage

Agent configuration is persisted:
- **Unique Agent ID** (UUID)
- **System Prompt** (generated)
- **Tool Bindings** (associated tool IDs)
- **Metadata** (creator, timestamps, tags)

#### Step 5: Agent Ready for Inference

The agent can now:
- Accept user queries via chat interface
- Execute using its inference engine (LangGraph state machine)
- Access bound tools
- Store conversation history

---

## Architecture Deep Dive

### Template Hierarchy

```
BaseAgentOnboard (Abstract)
├── ReactAgentOnboard
├── ReactCriticAgentOnboard
├── PlannerExecutorAgentOnboard
├── MultiAgentOnboard
├── HybridAgentOnboard
└── SimpleAIAgentOnboard

BaseMetaTypeAgentOnboard (Abstract)
├── MetaAgentOnboard
└── PlannerMetaAgentOnboard
```

**Key Distinction:**
- **BaseAgentOnboard**: For agents that use tools (functions/APIs)
- **BaseMetaTypeAgentOnboard**: For meta-agents that use other agents as "tools"

### Template Implementation Pattern

Every template follows this structure:

```python
class ReactAgentOnboard(BaseAgentOnboard):
    def __init__(self, agent_service_utils: AgentServiceUtils):
        # Set the agent type identifier
        super().__init__(agent_type="react_agent", agent_service_utils=agent_service_utils)

    async def _generate_system_prompt(self, agent_name, agent_goal,
                                      workflow_description, tool_or_worker_agents_prompt, llm):
        # Template-specific prompt generation logic
        # Uses LangChain PromptTemplate + LLM to generate context-aware system prompts
        react_system_prompt_template = PromptTemplate.from_template(react_system_prompt_generator)
        react_system_prompt_gen = react_system_prompt_template | llm | StrOutputParser()
        react_system_prompt = await react_system_prompt_gen.ainvoke({
            "agent_name": agent_name,
            "agent_goal": agent_goal,
            "workflow_description": workflow_description,
            "tool_prompt": tool_or_worker_agents_prompt
        })
        return {"SYSTEM_PROMPT_REACT_AGENT": react_system_prompt}
```

### Multi-Agent Template Example

The Multi-Agent template demonstrates advanced prompt generation using **LangGraph**:

```python
class MultiAgentOnboard(BaseAgentOnboard):
    async def _generate_system_prompt(self, agent_name, agent_goal,
                                      workflow_description, tool_or_worker_agents_prompt, llm):
        # Uses a LangGraph workflow to generate MULTIPLE system prompts in parallel
        agent_config_resp = await graph.ainvoke(input={
            'agent_name': agent_name,
            'agent_goal': agent_goal,
            'workflow_description': workflow_description,
            'tool_prompt': tool_or_worker_agents_prompt,
            'llm': llm
        })
        return agent_config_resp["MULTI_AGENT_SYSTEM_PROMPTS"]
```

**Generated Prompts for Multi-Agent:**
1. `SYSTEM_PROMPT_PLANNER_AGENT` - Creates execution plans
2. `SYSTEM_PROMPT_EXECUTOR_AGENT` - Executes the plan using tools
3. `SYSTEM_PROMPT_CRITIC_AGENT` - Evaluates execution quality
4. `SYSTEM_PROMPT_RESPONSE_GENERATOR_AGENT` - Formats final response
5. `SYSTEM_PROMPT_CRITIC_BASED_PLANNER_AGENT` - Replanning based on critique
6. `SYSTEM_PROMPT_REPLANNER_AGENT` - Adaptive replanning
7. `SYSTEM_PROMPT_GENERAL_LLM` - Handles general queries

All prompts are generated **in parallel** using LangGraph's state machine:

```
START
  ├──> ExecutorAgent ────┐
  ├──> PlannerAgent ─────┤
  ├──> CriticAgent ──────┤
  ├──> ... ──────────────┤
  └──> GeneralAgent ─────┴──> Merge ──> END
```

---

## Inference Engine

Each template has a corresponding **inference engine** that executes the agent logic using **LangGraph state machines**.

### Example: React Agent Inference Flow

```
User Input → Agent State → Reasoning Node → Tool Execution → Observation → Final Answer
                ↑                                                      │
                └──────────────────────────────────────────────────────┘
                             (Loop until task complete)
```

### Example: Multi-Agent Inference Flow

```
User Input → Query Classifier
              ├─> General Query → General LLM → Response
              └─> Goal-Specific Query → Planner → Executor → Critic
                                                      ↓
                                          Pass ← Evaluation? → Fail
                                            ↓                    ↓
                                        Response           Replanner → Executor
```

---

## Dependency Injection & Service Registry

### ServiceProvider Pattern

The framework uses dependency injection to provide template services:

```python
class ServiceProvider:
    @staticmethod
    def get_specialized_agent_service(agent_type: str) -> BaseAgentOnboard:
        if agent_type == "react_agent":
            return app_container.react_agent_service
        if agent_type == "multi_agent":
            return app_container.multi_agent_service
        # ... other templates
        raise HTTPException(status_code=400, detail=f"Unsupported agent type: {agent_type}")
```

**Initialization** (on application startup):

```python
# In app_container.py
self.react_agent_service = ReactAgentOnboard(agent_service_utils=self.agent_service_utils)
self.multi_agent_service = MultiAgentOnboard(agent_service_utils=self.agent_service_utils)
self.meta_agent_service = MetaAgentOnboard(agent_service_utils=self.agent_service_utils)
# ... all templates initialized once
```

This ensures:
- **Single instance** per template (singleton pattern)
- **Fast resolution** (no dynamic imports at request time)
- **Type safety** (returns concrete template types)

---

## Key Files Reference

| File Path | Purpose |
|-----------|---------|
| `src/agent_templates/base_agent_onboard.py` | Abstract base classes for templates |
| `src/agent_templates/react_agent_onboard.py` | React agent template implementation |
| `src/agent_templates/planner_executor_critic_agent_onboard.py` | Multi-agent template with parallel prompt generation |
| `src/agent_templates/meta_agent_onboard.py` | Meta-agent template for agent orchestration |
| `src/api/agent_endpoints.py` | REST API endpoints for agent CRUD operations |
| `src/api/dependencies.py` | ServiceProvider for dependency injection |
| `src/api/app_container.py` | Application container for service initialization |
| `src/schemas/agent_schemas.py` | Request/response schemas (AgentOnboardingRequest) |
| `src/prompts/prompts.py` | System prompt generation templates |
| `src/inference/` | Agent execution engines (LangGraph state machines) |
| `src/database/services.py` | AgentService, AgentServiceUtils for persistence |

---

## Adding a New Template

### Step 1: Create Template Class

```python
# src/agent_templates/my_agent_onboard.py
from src.agent_templates.base_agent_onboard import BaseAgentOnboard

class MyCustomAgentOnboard(BaseAgentOnboard):
    def __init__(self, agent_service_utils: AgentServiceUtils):
        super().__init__(agent_type="my_custom_agent", agent_service_utils=agent_service_utils)

    async def _generate_system_prompt(self, agent_name, agent_goal,
                                      workflow_description, tool_or_worker_agents_prompt, llm):
        # Custom prompt generation logic
        my_prompt_template = PromptTemplate.from_template(my_custom_prompt_generator)
        my_prompt_gen = my_prompt_template | llm | StrOutputParser()
        my_prompt = await my_prompt_gen.ainvoke({
            "agent_name": agent_name,
            "agent_goal": agent_goal,
            "workflow_description": workflow_description,
            "tool_prompt": tool_or_worker_agents_prompt
        })
        return {"SYSTEM_PROMPT_MY_AGENT": my_prompt}
```

### Step 2: Register in AppContainer

```python
# src/api/app_container.py
from src.agent_templates.my_agent_onboard import MyCustomAgentOnboard

class AppContainer:
    def __init__(self):
        self.my_custom_agent_service: MyCustomAgentOnboard = None

    async def initialize_services(self):
        # ... other initializations
        self.my_custom_agent_service = MyCustomAgentOnboard(agent_service_utils=self.agent_service_utils)
```

### Step 3: Add to ServiceProvider

```python
# src/api/dependencies.py
class ServiceProvider:
    @staticmethod
    def get_specialized_agent_service(agent_type: str) -> BaseAgentOnboard:
        # ... existing templates
        if agent_type == "my_custom_agent":
            return app_container.my_custom_agent_service
        raise HTTPException(status_code=400, detail=f"Unsupported agent type: {agent_type}")
```

### Step 4: Create Prompt Template

```python
# src/prompts/prompts.py
my_custom_prompt_generator = """
You are {agent_name}, specialized in {agent_goal}.

Follow this workflow:
{workflow_description}

Available tools:
{tool_prompt}

Your unique instruction: [Add custom behavior here]
"""
```

### Step 5: Create Inference Engine

```python
# src/inference/my_custom_agent_inference.py
from langgraph.graph import StateGraph, START, END

class MyCustomAgentInference(BaseAgentInference):
    async def _build_agent_graph(self, state_class, tools, system_prompts, llm):
        builder = StateGraph(state_class)
        # Define nodes and edges for execution logic
        builder.add_node("CustomNode", self.custom_logic)
        builder.add_edge(START, "CustomNode")
        builder.add_edge("CustomNode", END)
        return builder.compile()
```

---

## Template vs Agent: Key Distinction

| Aspect | Template | Agent |
|--------|----------|-------|
| **Definition** | A behavioral pattern class | An instance of a template |
| **Quantity** | 8 templates in the system | Unlimited agents can be created |
| **Customization** | Fixed code, defined by developers | Configured per user's needs |
| **System Prompt** | Template for generating prompts | Specific generated prompt |
| **Tools** | Accepts any compatible tools | Bound to specific tools |
| **Example** | `ReactAgentOnboard` class | "Customer Support Agent" (react_agent type) |

**Analogy**: Templates are like car models (Sedan, SUV, Truck), and agents are individual cars configured from those models (Red Sedan with GPS, Blue SUV with towing package).

---

## Enterprise Features Supporting Templates

### 1. RBAC (Role-Based Access Control)
- Admins can create/update/delete any agent
- Developers can create/update/delete their own agents
- Viewers can only view agents

### 2. Telemetry & Observability
- **OpenTelemetry** integration for distributed tracing
- **Arize Phoenix** for LLM observability
- Per-template telemetry projects (e.g., `onboard-react-agent`, `update-multi-agent`)

### 3. Agent Evaluation
- Ground-truth evaluation datasets
- LLM-as-a-Judge evaluation
- Per-agent and per-tool metrics

### 4. RAI Guardrails
- PII detection and masking
- Hallucination detection
- Bias mitigation
- Content moderation

### 5. Memory Management
- **Session Memory**: Conversation history within a session
- **Episodic Memory**: Long-term storage using embeddings
- **Semantic Search**: Cross-encoder for memory retrieval

### 6. Human-in-the-Loop
- Approval workflows for sensitive operations
- Agent action review and override
- Feedback collection

### 7. Export & Deployment
Agents can be exported as **standalone applications**:
- Dockerized backend (FastAPI)
- React frontend
- Bundled tools and dependencies
- Environment configuration
- Ready for production deployment

---

## Summary

The Infosys Agentic Foundry implements a sophisticated **template-based agent framework** where:

1. **Templates define agent behavior** (React, Multi-Agent, Meta, etc.)
2. **Agents are instantiated from templates** with specific goals and tools
3. **System prompts are dynamically generated** using LLMs based on template type
4. **Inference engines execute agents** using LangGraph state machines
5. **Dependency injection provides templates** via ServiceProvider
6. **Enterprise features** (RBAC, telemetry, evaluation, guardrails) ensure production readiness

This modular architecture enables:
- ✅ Rapid agent creation (minutes, not days)
- ✅ Consistent behavior patterns across use cases
- ✅ Easy customization without code changes
- ✅ Scalable multi-agent orchestration
- ✅ Enterprise-grade reliability and security

---

## Next Steps for Understanding

To deepen understanding of specific areas:

1. **Inference Engines**: Read `src/inference/react_agent_inference.py` to see LangGraph state machines in action
2. **Tool Integration**: Explore `src/tools/` to understand how tools are validated, formatted, and bound to agents
3. **Database Schema**: Review `src/database/repositories.py` to see how agents are persisted
4. **Frontend Integration**: Check `Infosys-Agentic-Foundry-Frontend/src/components/` to see the UI for agent creation
5. **Export Mechanism**: Examine `Export_Agent/AgentsExport.py` to understand agent packaging

---

**Document Version**: 1.0
**Last Updated**: 2025-12-29
**Repository**: Infosys-Agentic-Foundry
