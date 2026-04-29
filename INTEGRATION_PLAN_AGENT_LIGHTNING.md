# Integration Plan: Infosys Agentic Foundry + Microsoft Agent Lightning

## Executive Summary

This document outlines comprehensive integration strategies to combine **Infosys Agentic Foundry** (enterprise agent creation framework) with **Microsoft Agent Lightning** (RL training infrastructure) to create a unified platform that enables both rapid agent deployment AND continuous improvement through reinforcement learning.

**Key Value Proposition:**
- Create agents in minutes using Foundry templates
- Train and optimize them continuously using Agent Lightning RL
- Enterprise-grade deployment with production monitoring
- Close the feedback loop: Deploy → Collect Data → Train → Improve → Deploy

---

## Table of Contents

1. [Integration Architecture Overview](#1-integration-architecture-overview)
2. [Integration Routes (Multiple Options)](#2-integration-routes-multiple-options)
3. [Detailed Implementation Plans](#3-detailed-implementation-plans)
4. [Technical Design](#4-technical-design)
5. [Use Cases and Benefits](#5-use-cases-and-benefits)
6. [Challenges and Solutions](#6-challenges-and-solutions)
7. [Phased Rollout Strategy](#7-phased-rollout-strategy)
8. [Code Examples](#8-code-examples)

---

## 1. Integration Architecture Overview

### Current State

**Infosys Agentic Foundry:**
```
User → FastAPI → Template Selection → Prompt Generation → Database → LangGraph Inference
```

**Microsoft Agent Lightning:**
```
Algorithm → LightningStore → Runners → Agent Execution → Spans → Reward → Update
```

### Integrated Vision

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     INFOSYS AGENTIC FOUNDRY                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐               │
│  │   Agent      │   │  Template    │   │  Inference   │               │
│  │  Onboarding  │──▶│  Generation  │──▶│   Engine     │               │
│  └──────────────┘   └──────────────┘   └──────┬───────┘               │
│                                                 │                        │
│                                                 ▼                        │
│  ┌──────────────────────────────────────────────────────┐              │
│  │          RL TRAINING LAYER (Agent Lightning)         │              │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │              │
│  │  │ Lightning   │  │  Algorithm  │  │   Runners   │ │              │
│  │  │   Store     │◄─┤  (VERL/APO) │◄─┤  (Workers)  │ │              │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │              │
│  └──────────────────────────────────────────────────────┘              │
│                                                 │                        │
│                                                 ▼                        │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐               │
│  │  Evaluation  │   │  Telemetry   │   │   Export &   │               │
│  │   Service    │   │   (Phoenix)  │   │   Deploy     │               │
│  └──────────────┘   └──────────────┘   └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Integration Routes (Multiple Options)

### Route 1: Lightweight Integration (Recommended First Step)

**Approach:** Add Agent Lightning as an optional training module without modifying core Foundry architecture.

**Architecture:**
```
Foundry Agents (Unchanged) ──┐
                              ├──▶ Lightning Wrapper ──▶ Training Pipeline
Foundry Inference (Unchanged)─┘
```

**Pros:**
- Minimal code changes to Foundry
- Quick implementation (2-4 weeks)
- Low risk to existing system
- Easy to rollback

**Cons:**
- Some duplication (two inference paths)
- Manual coordination needed between systems

---

### Route 2: Deep Integration (Long-term Vision)

**Approach:** Embed Agent Lightning directly into Foundry's agent lifecycle.

**Architecture:**
```
Agent Creation → Template Gen → Training Loop → Optimized Agent → Production Deploy
                                    ↑                                      │
                                    └──────── Feedback Loop ───────────────┘
```

**Pros:**
- Seamless user experience
- Automatic training pipeline
- Unified monitoring and evaluation
- True closed-loop optimization

**Cons:**
- Significant refactoring required (2-3 months)
- Complex testing and validation
- Higher risk during migration

---

### Route 3: Hybrid Approach (Recommended Overall)

**Approach:** Start with Route 1, incrementally evolve toward Route 2.

**Phases:**
1. **Phase 1 (Month 1-2):** Lightweight integration + pilot projects
2. **Phase 2 (Month 3-4):** Integrate with Foundry's evaluation service
3. **Phase 3 (Month 5-6):** Embed into agent lifecycle
4. **Phase 4 (Month 7+):** Advanced features (multi-agent training, auto-optimization)

**Pros:**
- Balanced risk/reward
- Continuous delivery of value
- Allows learning and adjustment
- Production feedback informs design

**Cons:**
- Requires longer timeline
- Temporary technical debt during transition

---

## 3. Detailed Implementation Plans

### Route 1 Implementation: Lightweight Integration

#### Step 1: Create Lightning Adapter Service

**File:** `src/training/agent_lightning_adapter.py`

**Purpose:** Wrapper to convert Foundry agents into LitAgents

```python
from agentlightning import LitAgent, emit_reward
from src.inference.base_agent_inference import BaseAgentInference

class FoundryLitAgentAdapter(LitAgent[dict]):
    """Adapts Foundry agents to Agent Lightning interface"""

    def __init__(self,
                 agent_id: str,
                 agent_service: AgentService,
                 inference_service: BaseAgentInference,
                 reward_function: Callable):
        self.agent_id = agent_id
        self.agent_service = agent_service
        self.inference_service = inference_service
        self.reward_function = reward_function

    async def rollout(self, task: dict, resources: NamedResources, rollout: Rollout):
        """Execute single training rollout"""
        # 1. Get agent configuration
        agent_config = await self.agent_service.get_agent(self.agent_id)

        # 2. Update system prompt from resources (if provided)
        if "system_prompt" in resources:
            agent_config["system_prompt"] = resources["system_prompt"]

        # 3. Execute agent using Foundry's inference engine
        result = await self.inference_service.process(
            agent_config=agent_config,
            user_query=task["query"],
            session_id=rollout.rollout_id,
            # Lightning tracer automatically captures all LLM calls
        )

        # 4. Compute reward
        reward = self.reward_function(task, result)
        emit_reward(reward)

        # 5. Return result (spans auto-captured)
        return result
```

#### Step 2: Add Training Endpoints

**File:** `src/api/training_endpoints.py`

```python
from fastapi import APIRouter, Depends
from agentlightning import Trainer
from agentlightning.algorithm.verl import VERL
from agentlightning.algorithm.apo import APO

router = APIRouter(prefix="/training", tags=["Training"])

@router.post("/agents/{agent_id}/train")
async def train_agent(
    agent_id: str,
    training_request: AgentTrainingRequest,
    agent_service: AgentService = Depends(ServiceProvider.get_agent_service),
    authorization_service: AuthorizationService = Depends(ServiceProvider.get_authorization_service),
    user_data: User = Depends(get_current_user)
):
    """Start RL training for an agent"""

    # 1. Permission check
    if not await authorization_service.check_operation_permission(
        user_data.email, user_data.role, "execute", "agents"
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # 2. Get agent and inference service
    agent_config = await agent_service.get_agent(agent_id)
    inference_service = ServiceProvider.get_specialized_inference_service(
        agent_config["agentic_application_type"]
    )

    # 3. Create LitAgent adapter
    lit_agent = FoundryLitAgentAdapter(
        agent_id=agent_id,
        agent_service=agent_service,
        inference_service=inference_service,
        reward_function=training_request.reward_function
    )

    # 4. Configure training algorithm
    if training_request.algorithm == "verl":
        algorithm = VERL(training_request.verl_config)
    elif training_request.algorithm == "apo":
        algorithm = APO(training_request.apo_config)
    else:
        algorithm = None  # Baseline (just collect traces)

    # 5. Create trainer
    trainer = Trainer(
        algorithm=algorithm,
        n_runners=training_request.n_runners,
        strategy=ClientServerExecutionStrategy()
    )

    # 6. Start training (async background task)
    background_tasks.add_task(
        run_training,
        trainer=trainer,
        agent=lit_agent,
        dataset=training_request.dataset,
        agent_id=agent_id,
        user_id=user_data.email
    )

    return {
        "status": "training_started",
        "agent_id": agent_id,
        "training_id": str(uuid.uuid4())
    }
```

#### Step 3: Reward Function Registry

**File:** `src/training/reward_functions.py`

```python
from typing import Callable, Dict, Any

class RewardFunctionRegistry:
    """Registry of reward functions for different agent types"""

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable):
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        return cls._registry.get(name)


# Example reward functions

@RewardFunctionRegistry.register("task_completion")
def task_completion_reward(task: dict, result: dict) -> float:
    """Binary reward: 1 if task completed, 0 otherwise"""
    return 1.0 if result.get("status") == "completed" else 0.0


@RewardFunctionRegistry.register("sql_correctness")
async def sql_correctness_reward(task: dict, result: dict) -> float:
    """SQL execution correctness"""
    expected = task.get("expected_result")
    actual = result.get("sql_result")

    # Execute and compare
    if expected == actual:
        return 1.0
    return 0.0


@RewardFunctionRegistry.register("llm_as_judge")
async def llm_judge_reward(task: dict, result: dict) -> float:
    """Use LLM to evaluate response quality"""
    judge_prompt = f"""
    Task: {task['query']}
    Response: {result['response']}

    Rate the quality of this response on a scale of 0-1.
    Consider accuracy, completeness, and helpfulness.
    Return only a number between 0 and 1.
    """

    judge_response = await llm_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}]
    )

    return float(judge_response.choices[0].message.content.strip())


@RewardFunctionRegistry.register("multi_criteria")
def multi_criteria_reward(task: dict, result: dict) -> dict:
    """Multi-dimensional reward"""
    return {
        "accuracy": compute_accuracy(task, result),
        "efficiency": compute_efficiency(result),
        "safety": compute_safety(result),
    }
```

#### Step 4: Training Configuration Schema

**File:** `src/schemas/training_schemas.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AgentTrainingRequest(BaseModel):
    """Request schema for agent training"""
    algorithm: str = Field(..., description="Training algorithm: 'verl', 'apo', 'baseline'")
    dataset_id: str = Field(..., description="Training dataset ID")
    reward_function: str = Field(..., description="Reward function name from registry")
    n_runners: int = Field(4, description="Number of parallel workers")
    max_epochs: int = Field(10, description="Maximum training epochs")

    # Algorithm-specific configs
    verl_config: Optional[Dict[str, Any]] = Field(None, description="VERL/PPO configuration")
    apo_config: Optional[Dict[str, Any]] = Field(None, description="APO configuration")

    # Training metadata
    validation_dataset_id: Optional[str] = None
    checkpoint_frequency: int = Field(1, description="Save checkpoint every N epochs")
    early_stopping_patience: int = Field(3, description="Stop if no improvement for N epochs")


class TrainingDataset(BaseModel):
    """Training dataset schema"""
    dataset_id: str
    name: str
    tasks: List[Dict[str, Any]]  # List of training tasks
    created_by: str
    created_at: str

class TrainingCheckpoint(BaseModel):
    """Training checkpoint schema"""
    checkpoint_id: str
    agent_id: str
    epoch: int
    metrics: Dict[str, float]
    resources: Dict[str, Any]  # Optimized prompts, model weights, etc.
    created_at: str
```

#### Step 5: Database Schema for Training

**File:** `src/database/repositories/training_repository.py`

```sql
-- Training runs table
CREATE TABLE IF NOT EXISTS training_runs (
    training_run_id UUID PRIMARY KEY,
    agent_id UUID REFERENCES agentic_application_repository(agentic_application_id),
    algorithm VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,  -- 'running', 'completed', 'failed'
    dataset_id UUID NOT NULL,
    reward_function VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    metrics JSONB,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Training checkpoints table
CREATE TABLE IF NOT EXISTS training_checkpoints (
    checkpoint_id UUID PRIMARY KEY,
    training_run_id UUID REFERENCES training_runs(training_run_id),
    epoch INT NOT NULL,
    metrics JSONB NOT NULL,
    resources JSONB NOT NULL,  -- Optimized prompts, etc.
    created_at TIMESTAMP DEFAULT NOW()
);

-- Training datasets table
CREATE TABLE IF NOT EXISTS training_datasets (
    dataset_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tasks JSONB NOT NULL,  -- Array of training tasks
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Training rollouts (link to Lightning Store)
CREATE TABLE IF NOT EXISTS training_rollouts (
    rollout_id UUID PRIMARY KEY,
    training_run_id UUID REFERENCES training_runs(training_run_id),
    task_input JSONB NOT NULL,
    status VARCHAR(20) NOT NULL,
    reward FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### Route 2 Implementation: Deep Integration

#### Architecture Changes

**1. Extend AgentService with Training Methods**

```python
# src/database/services.py

class AgentService:
    # ... existing methods ...

    async def enable_training(self, agent_id: str, training_config: dict):
        """Enable RL training for an agent"""
        # Update agent metadata
        await self.agent_repo.update_agent_training_config(agent_id, training_config)

        # Initialize Lightning Store for this agent
        store = await self._create_agent_lightning_store(agent_id)

        return {"status": "training_enabled", "agent_id": agent_id}

    async def train_agent(self, agent_id: str, dataset: list, algorithm_config: dict):
        """Start training for an agent"""
        # Get agent configuration
        agent_config = await self.get_agent(agent_id)

        # Create LitAgent wrapper
        lit_agent = self._create_lit_agent_wrapper(agent_config)

        # Run training
        trainer = self._create_trainer(algorithm_config)
        results = await trainer.fit(lit_agent, dataset)

        # Update agent with optimized resources
        await self.update_agent_from_training(agent_id, results)

        return results
```

**2. Add Training Stage to Agent Lifecycle**

```
Current: Create → Deploy → Use
New:     Create → Train → Optimize → Deploy → Use → Collect Feedback → Retrain
```

**3. Unified Inference with Training**

```python
# src/inference/base_agent_inference.py

class BaseAgentInference:
    def __init__(self, ..., enable_training: bool = False):
        self.enable_training = enable_training
        if enable_training:
            self.tracer = AgentOpsTracer()  # Agent Lightning tracer

    async def process(self, ...):
        if self.enable_training:
            # Wrap execution in trace context
            with self.tracer.trace_context(rollout_id, attempt_id):
                result = await self._execute_agent(...)
                emit_reward(self._compute_reward(...))
        else:
            # Normal inference
            result = await self._execute_agent(...)

        return result
```

---

### Route 3 Implementation: Hybrid Approach

#### Phase 1: Pilot Integration (Weeks 1-4)

**Deliverables:**
1. ✅ Lightning adapter service (`agent_lightning_adapter.py`)
2. ✅ Basic training endpoint (`POST /training/agents/{id}/train`)
3. ✅ Reward function registry (3-5 initial functions)
4. ✅ Training with APO on 1-2 agents
5. ✅ Documentation and examples

**Success Criteria:**
- Successfully train at least 2 agents using APO
- Measure performance improvement (before/after metrics)
- Zero impact on existing production agents

#### Phase 2: Evaluation Integration (Weeks 5-8)

**Deliverables:**
1. ✅ Connect Lightning Store to Foundry's Evaluation Service
2. ✅ Automatic reward computation from evaluation datasets
3. ✅ Training dashboard in Foundry UI
4. ✅ VERL integration for full RL training
5. ✅ Multi-agent training support

**Success Criteria:**
- Train at least 5 agents using VERL
- Automated evaluation pipeline working
- Training metrics visible in dashboard

#### Phase 3: Lifecycle Integration (Weeks 9-16)

**Deliverables:**
1. ✅ "Train Agent" button in agent creation UI
2. ✅ Automatic training triggers (scheduled, performance-based)
3. ✅ Checkpoint management and rollback
4. ✅ A/B testing infrastructure (compare agent versions)
5. ✅ Production feedback loop integration

**Success Criteria:**
- 50%+ of new agents use training
- Measurable improvement in agent performance
- Production deployment of trained agents

#### Phase 4: Advanced Features (Weeks 17+)

**Deliverables:**
1. ✅ Multi-agent RL (training meta-agents with worker coordination)
2. ✅ Continuous learning (agents improve from production data)
3. ✅ Auto-optimization (agents self-tune parameters)
4. ✅ Custom algorithm support (users define training logic)

---

## 4. Technical Design

### 4.1 LightningStore Integration Options

#### Option A: Use MongoDB Store (Recommended)

**Pros:**
- Foundry already uses PostgreSQL, MongoDB adds persistence
- Lightning's MongoLightningStore is production-ready
- Scales horizontally
- Easy backup and recovery

**Cons:**
- Additional dependency (MongoDB)
- Operational overhead

**Implementation:**
```python
from agentlightning.store.mongo import MongoLightningStore

# In app_container.py initialization
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
lightning_store = MongoLightningStore(
    uri=mongo_uri,
    db_name="foundry_training"
)
```

#### Option B: Custom PostgreSQL Store

**Pros:**
- Reuses existing Foundry database
- No new dependencies
- Unified data management

**Cons:**
- Requires implementing LightningStore interface (~1000 lines)
- PostgreSQL not optimized for high-write workloads
- More complex testing

**Implementation:**
```python
from agentlightning import LightningStore

class PostgresLightningStore(LightningStore):
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def enqueue_rollout(self, input, resources_id, ...):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO training_rollouts
                (rollout_id, input, resources_id, status, ...)
                VALUES ($1, $2, $3, $4, ...)
            """, ...)

    # ... implement all required methods ...
```

#### Option C: Hybrid (MongoDB for traces, PostgreSQL for metadata)

**Pros:**
- Best of both worlds
- MongoDB handles high-volume traces
- PostgreSQL manages agent metadata

**Cons:**
- Most complex
- Data consistency challenges

---

### 4.2 Tracer Integration with Foundry Telemetry

**Current State:**
- Foundry uses OpenTelemetry + Arize Phoenix
- Agent Lightning uses AgentOps/OpenTelemetry tracers

**Integration Strategy:**

```python
# Custom tracer that bridges both systems
class FoundryLightningTracer(Tracer):
    def __init__(self, phoenix_tracer, agentops_tracer):
        self.phoenix_tracer = phoenix_tracer
        self.agentops_tracer = agentops_tracer

    def start_span(self, name, ...):
        # Send to both systems
        phoenix_span = self.phoenix_tracer.start_span(name, ...)
        agentops_span = self.agentops_tracer.start_span(name, ...)

        return CompositeSpan([phoenix_span, agentops_span])
```

**Benefits:**
- Unified observability
- Training data visible in Phoenix dashboard
- No duplicate instrumentation

---

### 4.3 Reward Function Design Patterns

#### Pattern 1: Task-Specific Evaluation

```python
@RewardFunctionRegistry.register("sql_execution")
async def sql_execution_reward(task: dict, result: dict) -> float:
    """Execute SQL and compare results"""
    expected_df = execute_sql(task["expected_sql"])
    actual_df = execute_sql(result["generated_sql"])

    if expected_df.equals(actual_df):
        return 1.0

    # Partial credit for similar results
    similarity = compute_dataframe_similarity(expected_df, actual_df)
    return similarity
```

#### Pattern 2: LLM-as-Judge with Rubric

```python
@RewardFunctionRegistry.register("customer_support_quality")
async def customer_support_quality(task: dict, result: dict) -> dict:
    """Multi-criteria evaluation using LLM judge"""
    judge_prompt = f"""
    Evaluate this customer support response:

    Customer Query: {task['query']}
    Agent Response: {result['response']}

    Rate on the following criteria (0-1 scale):
    1. Accuracy: Is the information correct?
    2. Completeness: Does it fully address the query?
    3. Empathy: Is the tone appropriate and empathetic?
    4. Clarity: Is the response clear and easy to understand?

    Return JSON: {{"accuracy": X, "completeness": Y, "empathy": Z, "clarity": W}}
    """

    judge_response = await llm_judge(judge_prompt)
    scores = json.loads(judge_response)

    # Weighted average
    total_score = (
        scores["accuracy"] * 0.4 +
        scores["completeness"] * 0.3 +
        scores["empathy"] * 0.2 +
        scores["clarity"] * 0.1
    )

    return {
        "total": total_score,
        **scores
    }
```

#### Pattern 3: Human-in-the-Loop

```python
@RewardFunctionRegistry.register("hitl_approval")
async def hitl_approval_reward(task: dict, result: dict) -> float:
    """Request human approval for reward"""
    # Send to approval queue
    approval_request = await approval_service.request_approval(
        agent_id=task["agent_id"],
        task=task,
        result=result
    )

    # Wait for human decision (with timeout)
    approval = await approval_service.wait_for_approval(
        approval_request.id,
        timeout=3600  # 1 hour
    )

    if approval.status == "approved":
        return 1.0
    elif approval.status == "rejected":
        return 0.0
    else:
        return 0.5  # Timeout/uncertain
```

#### Pattern 4: Foundry Evaluation Service Integration

```python
@RewardFunctionRegistry.register("foundry_eval")
async def foundry_evaluation_reward(task: dict, result: dict) -> float:
    """Use Foundry's existing evaluation service"""
    evaluation_service = ServiceProvider.get_evaluation_service()

    # Run evaluation using Foundry's ground-truth datasets
    eval_result = await evaluation_service.evaluate_response(
        agent_id=task["agent_id"],
        task_id=task["task_id"],
        response=result["response"]
    )

    # Extract reward from evaluation metrics
    reward = eval_result["metrics"]["overall_score"]
    return reward
```

---

### 4.4 Multi-Agent Training Design

For Foundry's Meta Agent and Planner-Meta Agent templates:

```python
class MetaAgentLightningAdapter(LitAgent[dict]):
    """Adapter for training meta-agents"""

    async def rollout(self, task: dict, resources: NamedResources, rollout: Rollout):
        # 1. Meta-agent decides which worker agents to use
        meta_agent_result = await self.meta_agent_inference.process(
            task=task,
            resources=resources
        )

        # 2. Execute worker agents (also traced)
        worker_results = []
        for worker_task in meta_agent_result["worker_tasks"]:
            worker_result = await self.execute_worker(worker_task)
            worker_results.append(worker_result)

        # 3. Meta-agent aggregates results
        final_result = await self.meta_agent_inference.aggregate(worker_results)

        # 4. Compute hierarchical rewards
        worker_rewards = [self.reward_fn(task, r) for r in worker_results]
        meta_reward = self.reward_fn(task, final_result)

        # Emit rewards
        emit_reward({
            "meta_agent": meta_reward,
            "worker_agents": worker_rewards,
            "combined": (meta_reward + sum(worker_rewards)) / (len(worker_rewards) + 1)
        }, primary_key="combined")

        return final_result
```

---

## 5. Use Cases and Benefits

### Use Case 1: SQL Agent Optimization

**Scenario:** Improve Text-to-SQL agent accuracy from 60% → 85%

**Current Foundry Setup:**
- React Agent template with SQL generation tools
- Static system prompt
- Manual prompt engineering

**With Agent Lightning:**
1. Create training dataset from Foundry's evaluation logs
2. Use VERL/PPO to optimize SQL generation
3. Reward = SQL execution correctness
4. Train for 10 epochs with 1000 examples
5. Deploy optimized agent

**Expected Improvement:**
- Accuracy: 60% → 85% (+25pp)
- Training time: 2-4 hours
- Cost: $50-100 in compute

---

### Use Case 2: Customer Support Agent Prompt Optimization

**Scenario:** Optimize support agent prompts without RL (lighter approach)

**Current Foundry Setup:**
- Multi-Agent template (Planner-Executor-Critic)
- 7 system prompts (planner, executor, critic, etc.)
- Manual tuning

**With Agent Lightning (APO):**
1. Start with baseline prompts
2. Use APO to optimize each prompt independently
3. Reward = LLM-as-judge quality score
4. Beam search over prompt space
5. Deploy best prompts

**Expected Improvement:**
- Customer satisfaction: 3.5/5 → 4.2/5
- Response quality: +30%
- Training time: 4-6 hours
- Cost: $20-40 (no GPU needed)

---

### Use Case 3: Meta-Agent Worker Selection

**Scenario:** Train meta-agent to select optimal worker agents

**Current Foundry Setup:**
- Meta Agent with 5 worker agents
- Manual routing logic
- Static delegation rules

**With Agent Lightning:**
1. Collect historical data on worker performance
2. Train meta-agent to predict best worker for each task
3. Reward = worker success rate
4. Use multi-agent RL (meta + workers jointly optimized)

**Expected Improvement:**
- Task completion rate: 70% → 90%
- Efficiency: 40% fewer worker calls
- Training time: 8-12 hours

---

### Use Case 4: Continuous Learning from Production

**Scenario:** Agent improves automatically from user feedback

**Workflow:**
1. Agent deployed to production
2. Users rate responses (👍/👎)
3. Lightning Store collects traces + ratings as rewards
4. Nightly training job optimizes agent
5. Best checkpoint auto-deployed next day

**Benefits:**
- Zero manual intervention
- Continuous improvement
- Adapts to changing user needs
- Catches edge cases automatically

---

## 6. Challenges and Solutions

### Challenge 1: Token ID Preservation

**Problem:** RL training requires raw token IDs, but Foundry's LLM calls may only return text.

**Solutions:**
1. **Use vLLM for inference:** Agent Lightning's LLM Proxy supports vLLM
2. **Patch LLM clients:** Modify Foundry's model service to preserve token IDs
3. **Use Lightning's retokenization:** Adapter can retokenize text (less accurate)

**Recommended:** Option 1 (vLLM) for production, Option 3 for quick pilot.

---

### Challenge 2: Training Infrastructure Scaling

**Problem:** Training can require significant compute (GPUs for VERL).

**Solutions:**
1. **Cloud bursting:** Spin up GPU instances on-demand (AWS/Azure)
2. **Training queue:** Queue training jobs, run during off-peak hours
3. **APO first:** Start with gradient-free APO (no GPU needed)

**Recommended:** Phase 1 uses APO, Phase 2+ adds VERL with cloud GPUs.

---

### Challenge 3: Agent State Management

**Problem:** Foundry agents have complex state (memory, context, tools). How to handle during training?

**Solutions:**
1. **Stateless rollouts:** Each training task is independent (reset state)
2. **Episode-based training:** Group related tasks into episodes
3. **Shared memory:** Workers share episodic memory (advanced)

**Recommended:** Option 1 for simplicity, Option 2 for conversational agents.

---

### Challenge 4: Reward Function Quality

**Problem:** Poor reward functions lead to poor training (reward hacking, misalignment).

**Solutions:**
1. **Multi-criteria rewards:** Use multiple reward signals
2. **Human validation:** Sample-check rewards for quality
3. **Reward modeling:** Train separate reward model from human feedback (RLHF)
4. **Use Foundry's evaluation:** Leverage existing evaluation datasets

**Recommended:** Start with Option 4, add Option 1 for complex agents.

---

### Challenge 5: Integration with RBAC

**Problem:** Who can train agents? How to manage training permissions?

**Solutions:**
1. **Extend RBAC:** Add "train_agents" permission
2. **Training approval:** Admins must approve training jobs
3. **Resource limits:** Limit compute per user/role

**Implementation:**
```python
# In authorization_service.py
PERMISSIONS = {
    UserRole.ADMIN: [..., "train_agents", "approve_training"],
    UserRole.DEVELOPER: [..., "train_agents"],
    UserRole.VIEWER: [...],  # No training access
}
```

---

### Challenge 6: Multi-Turn Conversations

**Problem:** Foundry agents support multi-turn chats. How to train on conversations?

**Solutions:**
1. **Trajectory aggregation:** Agent Lightning's experimental feature
2. **Per-turn rewards:** Train on each turn independently
3. **Episode rewards:** Single reward for entire conversation

**Configuration:**
```python
# Enable trajectory-level training
verl_config["agentlightning"]["trace_aggregator"] = {
    "level": "trajectory",
    "trajectory_max_prompt_length": 4096,
}
```

**Recommended:** Start with Option 2, migrate to Option 1 for efficiency.

---

## 7. Phased Rollout Strategy

### Phase 1: Proof of Concept (Weeks 1-4)

**Goals:**
- Validate technical feasibility
- Train 2-3 agents successfully
- Identify integration challenges

**Tasks:**
1. ✅ Set up Agent Lightning in separate environment
2. ✅ Create FoundryLitAgentAdapter
3. ✅ Train React Agent on SQL task (APO)
4. ✅ Train Multi-Agent on support task (APO)
5. ✅ Measure performance improvements
6. ✅ Document learnings

**Success Metrics:**
- At least 1 agent shows >10% improvement
- Training completes without errors
- Zero impact on production

**Go/No-Go Decision:** If success metrics met, proceed to Phase 2.

---

### Phase 2: Limited Production (Weeks 5-12)

**Goals:**
- Deploy training capability to production
- Train 10-20 agents
- Gather user feedback

**Tasks:**
1. ✅ Implement training endpoints in Foundry API
2. ✅ Add MongoDB Lightning Store
3. ✅ Create training UI in Foundry frontend
4. ✅ Integrate with evaluation service
5. ✅ Add VERL support (GPU training)
6. ✅ Train 10 agents across different templates
7. ✅ A/B test trained vs. baseline agents

**Success Metrics:**
- 10+ agents trained
- Average >15% performance improvement
- User satisfaction with training UI >4/5
- No production incidents

**Go/No-Go Decision:** If success metrics met, proceed to Phase 3.

---

### Phase 3: Full Integration (Weeks 13-24)

**Goals:**
- Make training a core feature
- Achieve 50%+ adoption
- Automate training workflows

**Tasks:**
1. ✅ Deep integration with agent lifecycle
2. ✅ Automatic training triggers
3. ✅ Continuous learning from production
4. ✅ Multi-agent training support
5. ✅ Custom algorithm framework
6. ✅ Advanced UI (training dashboard, checkpoint management)
7. ✅ Documentation and training materials

**Success Metrics:**
- 50%+ of new agents use training
- Average 20%+ performance improvement
- 5+ custom algorithms developed by users
- Zero critical bugs

---

### Phase 4: Innovation (Weeks 25+)

**Goals:**
- Push boundaries of agent training
- Research new algorithms
- Build competitive moat

**Tasks:**
1. ✅ Multi-agent RL (joint optimization)
2. ✅ Meta-learning (agents learn to learn)
3. ✅ Transfer learning (reuse across agents)
4. ✅ Reinforcement learning from human feedback (RLHF)
5. ✅ Auto-optimization (self-improving agents)
6. ✅ Research partnerships (publish papers)

---

## 8. Code Examples

### Example 1: Training a React Agent with APO

```python
from agentlightning import Trainer
from agentlightning.algorithm.apo import APO
from src.training.agent_lightning_adapter import FoundryLitAgentAdapter
from src.training.reward_functions import RewardFunctionRegistry

# 1. Load agent from Foundry
agent_id = "react-agent-sql-123"
agent_service = ServiceProvider.get_agent_service()
agent_config = await agent_service.get_agent(agent_id)

# 2. Create training dataset
train_dataset = [
    {
        "query": "Find all customers from California",
        "expected_sql": "SELECT * FROM customers WHERE state = 'CA'",
        "db_schema": {...}
    },
    # ... more examples
]

# 3. Create LitAgent adapter
lit_agent = FoundryLitAgentAdapter(
    agent_id=agent_id,
    agent_service=agent_service,
    inference_service=ServiceProvider.get_specialized_inference_service("react_agent"),
    reward_function=RewardFunctionRegistry.get("sql_correctness")
)

# 4. Configure APO
apo_config = {
    "n_iterations": 5,
    "beam_width": 3,
    "n_rollouts_per_prompt": 10,
}

# 5. Create trainer
trainer = Trainer(
    algorithm=APO(apo_config),
    n_runners=4
)

# 6. Train
results = await trainer.fit(lit_agent, train_dataset)

# 7. Update agent with best prompt
best_prompt = results["best_resources"]["system_prompt"]
await agent_service.update_agent(
    agentic_application_id=agent_id,
    system_prompt={"SYSTEM_PROMPT_REACT_AGENT": best_prompt}
)

print(f"Training complete! Improvement: {results['improvement_pct']}%")
```

---

### Example 2: Training a Multi-Agent with VERL

```python
from agentlightning import Trainer
from agentlightning.algorithm.verl import VERL
from src.training.agent_lightning_adapter import FoundryLitAgentAdapter

# 1. Load multi-agent
agent_id = "multi-agent-support-456"
agent_config = await agent_service.get_agent(agent_id)

# 2. Create adapter (multi-agent has multiple system prompts)
class MultiAgentAdapter(FoundryLitAgentAdapter):
    async def rollout(self, task, resources, rollout):
        # Update ALL system prompts from resources
        for key in ["SYSTEM_PROMPT_PLANNER_AGENT", "SYSTEM_PROMPT_EXECUTOR_AGENT", ...]:
            if key in resources:
                agent_config["system_prompt"][key] = resources[key]

        # Execute multi-agent workflow
        result = await self.inference_service.process(agent_config, task["query"])

        # Compute reward
        reward = await llm_judge_reward(task, result)
        emit_reward(reward)

        return result

# 3. Configure VERL
verl_config = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_batch_size": 32,
        "max_prompt_length": 4096,
        "max_response_length": 2048,
    },
    "actor_rollout_ref": {
        "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
        "actor": {
            "ppo_mini_batch_size": 8,
            "ppo_epochs": 2,
        },
    },
    "trainer": {
        "total_epochs": 10,
        "save_freq": 1,
    },
}

# 4. Train with VERL (requires GPU)
trainer = Trainer(
    algorithm=VERL(verl_config),
    n_runners=8,  # More workers for GPU utilization
)

results = await trainer.fit(MultiAgentAdapter(...), train_dataset, val_dataset)

# 5. Update all prompts
for key, value in results["best_resources"].items():
    if key.startswith("SYSTEM_PROMPT_"):
        agent_config["system_prompt"][key] = value

await agent_service.update_agent(agent_id, system_prompt=agent_config["system_prompt"])
```

---

### Example 3: Continuous Learning from Production

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Set up nightly training job
scheduler = AsyncIOScheduler()

async def nightly_training_job():
    """Train agents on production data collected during the day"""

    # 1. Get all agents with training enabled
    agents_to_train = await agent_service.get_agents_with_training_enabled()

    for agent_config in agents_to_train:
        agent_id = agent_config["agentic_application_id"]

        # 2. Collect production data from last 24 hours
        production_data = await chat_service.get_recent_conversations(
            agent_id=agent_id,
            hours=24,
            min_reward=None  # Include all conversations
        )

        # 3. Filter for quality (only use highly-rated interactions)
        train_data = [
            conv for conv in production_data
            if conv.get("user_rating", 0) >= 4  # 4+ stars
        ]

        if len(train_data) < 10:
            log.info(f"Insufficient training data for {agent_id}, skipping")
            continue

        # 4. Create training dataset
        dataset = [
            {
                "query": conv["user_query"],
                "expected_response": conv["agent_response"],
                "reward": conv["user_rating"] / 5.0  # Normalize to 0-1
            }
            for conv in train_data
        ]

        # 5. Train with APO (lightweight, no GPU)
        lit_agent = FoundryLitAgentAdapter(
            agent_id=agent_id,
            agent_service=agent_service,
            inference_service=ServiceProvider.get_specialized_inference_service(
                agent_config["agentic_application_type"]
            ),
            reward_function=lambda task, result: task["reward"]  # Use production ratings
        )

        trainer = Trainer(algorithm=APO({"n_iterations": 3, "beam_width": 2}))
        results = await trainer.fit(lit_agent, dataset)

        # 6. Deploy if improvement detected
        if results["improvement_pct"] > 5:  # At least 5% better
            await agent_service.update_agent_from_training(agent_id, results)
            log.info(f"Updated {agent_id} with {results['improvement_pct']}% improvement")
        else:
            log.info(f"No significant improvement for {agent_id}, keeping current version")

# Schedule for 2 AM daily
scheduler.add_job(nightly_training_job, 'cron', hour=2)
scheduler.start()
```

---

## 9. Migration Path from Existing Foundry Setup

### For Existing Agents

**Option 1: Opt-in (Recommended)**
- Agents continue working as-is
- Users explicitly enable training per agent
- Zero breaking changes

**Option 2: Automatic Enablement**
- All agents get training capability by default
- Users can opt-out if desired
- Requires careful communication

**Recommended Approach:**
```python
# Add "training_enabled" flag to agent schema
class AgentOnboardingRequest(BaseModel):
    # ... existing fields ...
    enable_training: bool = Field(False, description="Enable RL training for this agent")
    training_config: Optional[TrainingConfig] = None

# Only enable Lightning features if training_enabled=True
if agent_config.get("enable_training"):
    inference_service = TrainingEnabledInference(...)
else:
    inference_service = StandardInference(...)
```

---

## 10. Success Metrics and KPIs

### Technical Metrics

1. **Training Success Rate:** % of training jobs that complete successfully
   - Target: >95%

2. **Performance Improvement:** Average improvement after training
   - Target: >15% on task-specific metrics

3. **Training Time:** Average time to train an agent
   - Target: <4 hours for APO, <12 hours for VERL

4. **System Stability:** Impact on production system
   - Target: Zero critical incidents, <1% latency increase

### Business Metrics

1. **Adoption Rate:** % of agents using training
   - Phase 1: 10%
   - Phase 2: 30%
   - Phase 3: 50%+

2. **User Satisfaction:** Rating of training feature
   - Target: >4.0/5.0

3. **ROI:** Value delivered vs. infrastructure cost
   - Target: 10x ROI (improved agent performance vs. training costs)

4. **Time to Optimal Agent:** Time from creation to production-ready
   - Current: 2-4 weeks (manual iteration)
   - Target: 1-3 days (automated training)

---

## 11. Risk Mitigation

### Risk 1: Training Degrades Agent Performance

**Mitigation:**
- Always maintain baseline checkpoint
- A/B test before deployment
- Automated rollback if metrics decline
- Manual approval for production deployment

### Risk 2: Training Infrastructure Failures

**Mitigation:**
- Graceful degradation (agents work without training)
- Retry logic for transient failures
- Monitoring and alerting
- Regular backups of checkpoints

### Risk 3: Security and Privacy

**Mitigation:**
- Training data access control (RBAC)
- PII detection before training
- Encrypted storage for training data
- Audit logs for all training activities

### Risk 4: Cost Overruns

**Mitigation:**
- Budget limits per user/organization
- Auto-shutdown for long-running jobs
- Use APO (cheaper) before VERL
- Cloud cost monitoring and alerts

---

## Conclusion

Integrating Microsoft Agent Lightning with Infosys Agentic Foundry creates a powerful closed-loop system:

1. **Create** agents rapidly using Foundry templates
2. **Train** agents automatically using Lightning RL
3. **Deploy** optimized agents to production
4. **Collect** feedback and performance data
5. **Improve** continuously through automated retraining

This integration transforms Foundry from a deployment platform into a **continuous improvement platform**, enabling organizations to build agents that get better over time.

**Recommended Starting Point:** Route 3 (Hybrid Approach), Phase 1 (Pilot), using APO for initial validation. This minimizes risk while delivering immediate value.

---

**Next Steps:**
1. Review this plan with stakeholders
2. Select integration route and phase
3. Set up development environment
4. Begin Phase 1 implementation
5. Iterate based on learnings

**Estimated Timeline:**
- Phase 1 (Pilot): 4 weeks
- Phase 2 (Production): 8 weeks
- Phase 3 (Full Integration): 12 weeks
- **Total**: 6 months to production-ready integrated system

**Estimated Investment:**
- Engineering: 2-3 senior engineers
- Infrastructure: $500-1000/month (MongoDB, GPU instances)
- Total: ~$100k for 6-month integration

**Expected Return:**
- 20%+ improvement in agent performance
- 50% reduction in manual prompt engineering
- Continuous learning from production data
- Competitive differentiation in market
