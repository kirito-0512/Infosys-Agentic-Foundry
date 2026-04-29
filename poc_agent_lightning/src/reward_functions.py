# © 2024-25 Infosys Limited, Bangalore, India. All Rights Reserved.
"""
Reward Function Registry for Agent Lightning training.

This module provides a registry of reward functions that can be used to evaluate
agent performance during training. Reward functions take a task and result as input
and return a scalar or dictionary of rewards.
"""

from typing import Callable, Dict, Any, Optional
import asyncio
import pandas as pd
from telemetry_wrapper import logger as log


class RewardFunctionRegistry:
    """
    Registry of reward functions for different agent training scenarios.

    Reward functions are registered with a name and can be retrieved by name
    for use in training configurations.
    """

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a reward function.

        Example:
            @RewardFunctionRegistry.register("my_reward")
            def my_reward_function(task: dict, result: dict) -> float:
                return 1.0 if result["success"] else 0.0
        """
        def decorator(func: Callable):
            cls._registry[name] = func
            log.info(f"Registered reward function: {name}")
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """Retrieve a reward function by name."""
        return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered reward function names."""
        return list(cls._registry.keys())


# ============================================================================
# Basic Reward Functions
# ============================================================================

@RewardFunctionRegistry.register("task_completion")
def task_completion_reward(task: Dict[str, Any], result: Dict[str, Any]) -> float:
    """
    Binary reward: 1.0 if task completed successfully, 0.0 otherwise.

    Args:
        task: Training task
        result: Agent execution result with "status" field

    Returns:
        1.0 if status is "completed" or "success", else 0.0
    """
    status = result.get("status", "").lower()
    return 1.0 if status in ["completed", "success", "succeeded"] else 0.0


@RewardFunctionRegistry.register("exact_match")
def exact_match_reward(task: Dict[str, Any], result: Dict[str, Any]) -> float:
    """
    Exact match reward: 1.0 if output exactly matches expected, else 0.0.

    Args:
        task: Training task with "expected_output" field
        result: Agent execution result with "response" or "output" field

    Returns:
        1.0 if exact match, else 0.0
    """
    expected = task.get("expected_output", "").strip()
    actual = result.get("response", result.get("output", "")).strip()
    return 1.0 if expected == actual else 0.0


# ============================================================================
# SQL-Specific Reward Functions
# ============================================================================

@RewardFunctionRegistry.register("sql_correctness")
async def sql_correctness_reward(task: Dict[str, Any], result: Dict[str, Any]) -> float:
    """
    SQL execution correctness reward.

    Executes both expected and generated SQL, compares results.
    Awards 1.0 for exact match, partial credit for similarity.

    Args:
        task: Training task with:
            - expected_sql: Expected SQL query
            - db_connection: Database connection info
        result: Agent result with:
            - generated_sql: Generated SQL query

    Returns:
        0.0-1.0 based on result similarity
    """
    try:
        expected_sql = task.get("expected_sql", "")
        generated_sql = result.get("generated_sql", result.get("response", ""))

        # TODO: Integrate with Foundry's database connection manager
        # For POC, using simple comparison
        if expected_sql.strip().lower() == generated_sql.strip().lower():
            return 1.0

        # Partial credit for structural similarity
        # (In production, execute and compare results)
        similarity = _compute_sql_similarity(expected_sql, generated_sql)
        return similarity

    except Exception as e:
        log.error(f"Error computing SQL correctness: {str(e)}")
        return 0.0


def _compute_sql_similarity(expected: str, generated: str) -> float:
    """Compute structural similarity between SQL queries."""
    # Simple token-based similarity (improve in production)
    expected_tokens = set(expected.lower().split())
    generated_tokens = set(generated.lower().split())

    if not expected_tokens:
        return 0.0

    intersection = len(expected_tokens & generated_tokens)
    union = len(expected_tokens | generated_tokens)

    return intersection / union if union > 0 else 0.0


@RewardFunctionRegistry.register("sql_execution")
async def sql_execution_reward(task: Dict[str, Any], result: Dict[str, Any]) -> float:
    """
    SQL execution reward with actual database execution.

    Executes both queries and compares DataFrames.

    Args:
        task: Training task with database connection and expected SQL
        result: Agent result with generated SQL

    Returns:
        1.0 for exact match, 0.0-1.0 for partial match
    """
    try:
        from src.api.dependencies import ServiceProvider

        # Get database connection manager
        db_manager = ServiceProvider.get_multi_db_connection_manager()

        # Execute expected SQL
        expected_sql = task.get("expected_sql", "")
        db_config = task.get("db_config", {})
        expected_df = await _execute_sql(db_manager, expected_sql, db_config)

        # Execute generated SQL
        generated_sql = result.get("generated_sql", result.get("response", ""))
        actual_df = await _execute_sql(db_manager, generated_sql, db_config)

        # Compare results
        if expected_df.equals(actual_df):
            return 1.0

        # Partial credit for similar results
        similarity = _compute_dataframe_similarity(expected_df, actual_df)
        return similarity

    except Exception as e:
        log.error(f"Error executing SQL: {str(e)}")
        return 0.0


async def _execute_sql(db_manager, sql: str, db_config: dict) -> pd.DataFrame:
    """Execute SQL and return DataFrame."""
    # TODO: Implement using Foundry's database connection manager
    # Placeholder for POC
    return pd.DataFrame()


def _compute_dataframe_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Compute similarity between two DataFrames."""
    if df1.shape != df2.shape:
        # Different shapes: partial credit based on size similarity
        size_sim = min(df1.size, df2.size) / max(df1.size, df2.size)
        return size_sim * 0.5

    # Same shape: compute cell-wise match
    matches = (df1 == df2).sum().sum()
    total = df1.size
    return matches / total if total > 0 else 0.0


# ============================================================================
# LLM-as-Judge Reward Functions
# ============================================================================

@RewardFunctionRegistry.register("llm_judge_quality")
async def llm_judge_quality_reward(task: Dict[str, Any], result: Dict[str, Any]) -> float:
    """
    LLM-as-judge quality evaluation.

    Uses a separate LLM to evaluate response quality on a 0-1 scale.

    Args:
        task: Training task with "query" field
        result: Agent result with "response" field

    Returns:
        0.0-1.0 quality score from judge LLM
    """
    try:
        from src.models.model_service import ModelService
        from src.api.dependencies import ServiceProvider

        model_service = ServiceProvider.get_model_service()

        query = task.get("query", "")
        response = result.get("response", "")

        judge_prompt = f"""
Evaluate the quality of this AI assistant response.

User Query: {query}

Assistant Response: {response}

Rate the response quality on a scale of 0.0 to 1.0, considering:
- Accuracy: Is the information correct?
- Completeness: Does it fully address the query?
- Clarity: Is it clear and easy to understand?
- Helpfulness: Does it help the user?

Return ONLY a number between 0.0 and 1.0, nothing else.
"""

        # Call judge LLM
        judge_response = await model_service.generate_completion(
            model_name="gpt-4",  # Use strong model for judging
            prompt=judge_prompt,
            temperature=0.0  # Deterministic judging
        )

        # Parse score
        score_str = judge_response.strip()
        score = float(score_str)

        # Clamp to 0-1
        return max(0.0, min(1.0, score))

    except Exception as e:
        log.error(f"Error in LLM judge: {str(e)}")
        return 0.5  # Neutral score on error


@RewardFunctionRegistry.register("llm_judge_multi_criteria")
async def llm_judge_multi_criteria_reward(
    task: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, float]:
    """
    Multi-criteria LLM-as-judge evaluation.

    Returns separate scores for different quality dimensions.

    Args:
        task: Training task
        result: Agent result

    Returns:
        Dictionary with scores for accuracy, completeness, empathy, clarity
    """
    try:
        from src.models.model_service import ModelService
        from src.api.dependencies import ServiceProvider
        import json

        model_service = ServiceProvider.get_model_service()

        query = task.get("query", "")
        response = result.get("response", "")

        judge_prompt = f"""
Evaluate this AI assistant response on multiple criteria.

User Query: {query}

Assistant Response: {response}

Rate the response on these criteria (0.0-1.0 scale):
1. Accuracy: Is the information correct?
2. Completeness: Does it fully address the query?
3. Empathy: Is the tone appropriate and empathetic?
4. Clarity: Is the response clear and easy to understand?

Return ONLY a JSON object with these exact keys:
{{"accuracy": 0.0-1.0, "completeness": 0.0-1.0, "empathy": 0.0-1.0, "clarity": 0.0-1.0}}
"""

        judge_response = await model_service.generate_completion(
            model_name="gpt-4",
            prompt=judge_prompt,
            temperature=0.0
        )

        # Parse JSON
        scores = json.loads(judge_response.strip())

        # Compute weighted total
        total_score = (
            scores["accuracy"] * 0.4 +
            scores["completeness"] * 0.3 +
            scores["empathy"] * 0.2 +
            scores["clarity"] * 0.1
        )

        return {
            "total": total_score,
            "accuracy": scores["accuracy"],
            "completeness": scores["completeness"],
            "empathy": scores["empathy"],
            "clarity": scores["clarity"],
            "_primary": "total"  # Indicates which key to use for optimization
        }

    except Exception as e:
        log.error(f"Error in multi-criteria judge: {str(e)}")
        return {
            "total": 0.5,
            "accuracy": 0.5,
            "completeness": 0.5,
            "empathy": 0.5,
            "clarity": 0.5,
            "_primary": "total"
        }


# ============================================================================
# Foundry Evaluation Integration
# ============================================================================

@RewardFunctionRegistry.register("foundry_evaluation")
async def foundry_evaluation_reward(task: Dict[str, Any], result: Dict[str, Any]) -> float:
    """
    Integrate with Foundry's existing evaluation service.

    Uses ground-truth datasets and evaluation metrics already configured in Foundry.

    Args:
        task: Training task with:
            - agent_id: Agent being trained
            - task_id: Evaluation task ID
        result: Agent result with response

    Returns:
        Evaluation score from Foundry's evaluation service
    """
    try:
        from src.api.dependencies import ServiceProvider

        evaluation_service = ServiceProvider.get_evaluation_service()

        # Run evaluation using Foundry's service
        eval_result = await evaluation_service.evaluate_response(
            agent_id=task.get("agent_id"),
            task_id=task.get("task_id"),
            response=result.get("response", "")
        )

        # Extract overall score
        reward = eval_result.get("metrics", {}).get("overall_score", 0.0)
        return reward

    except Exception as e:
        log.error(f"Error in Foundry evaluation: {str(e)}")
        return 0.0


# ============================================================================
# Human-in-the-Loop Reward Functions
# ============================================================================

@RewardFunctionRegistry.register("hitl_approval")
async def hitl_approval_reward(task: Dict[str, Any], result: Dict[str, Any]) -> float:
    """
    Human-in-the-loop approval reward.

    Sends result to human approver and waits for decision.
    Use sparingly as it blocks training!

    Args:
        task: Training task
        result: Agent result to approve

    Returns:
        1.0 if approved, 0.0 if rejected, 0.5 if timeout
    """
    try:
        from src.api.dependencies import ServiceProvider

        authorization_service = ServiceProvider.get_authorization_service()

        # Request human approval
        approval_request = await authorization_service.request_approval(
            agent_id=task.get("agent_id"),
            task=task,
            result=result,
            timeout=3600  # 1 hour timeout
        )

        # Wait for decision (async, non-blocking for other rollouts)
        approval = await authorization_service.wait_for_approval(
            approval_request.id,
            timeout=3600
        )

        if approval.status == "approved":
            return 1.0
        elif approval.status == "rejected":
            return 0.0
        else:
            return 0.5  # Timeout or uncertain

    except Exception as e:
        log.error(f"Error in HITL approval: {str(e)}")
        return 0.5


# ============================================================================
# Custom Domain-Specific Rewards
# ============================================================================

@RewardFunctionRegistry.register("customer_support_quality")
async def customer_support_quality_reward(
    task: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, float]:
    """
    Customer support specific quality metrics.

    Evaluates:
    - Issue resolution (did it solve the problem?)
    - Response time (how quickly?)
    - Empathy score (tone analysis)
    - Escalation appropriateness (should it escalate?)

    Returns:
        Multi-dimensional reward
    """
    # Use LLM judge for detailed evaluation
    return await llm_judge_multi_criteria_reward(task, result)


@RewardFunctionRegistry.register("code_generation_quality")
async def code_generation_quality_reward(
    task: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, float]:
    """
    Code generation quality metrics.

    Evaluates:
    - Correctness (does it work?)
    - Test coverage (are there tests?)
    - Code quality (is it clean?)
    - Documentation (is it documented?)

    Returns:
        Multi-dimensional reward
    """
    try:
        generated_code = result.get("generated_code", result.get("response", ""))
        test_cases = task.get("test_cases", [])

        # Execute tests
        correctness = await _run_code_tests(generated_code, test_cases)

        # Static analysis (simplified)
        quality = _analyze_code_quality(generated_code)

        return {
            "total": correctness * 0.7 + quality * 0.3,
            "correctness": correctness,
            "quality": quality,
            "_primary": "total"
        }

    except Exception as e:
        log.error(f"Error evaluating code: {str(e)}")
        return {"total": 0.0, "correctness": 0.0, "quality": 0.0, "_primary": "total"}


async def _run_code_tests(code: str, test_cases: list) -> float:
    """Run test cases against generated code."""
    # TODO: Implement safe code execution sandbox
    # Placeholder for POC
    return 0.5


def _analyze_code_quality(code: str) -> float:
    """Analyze code quality (complexity, style, etc.)."""
    # TODO: Integrate pylint, flake8, or similar
    # Placeholder for POC
    return 0.5


# ============================================================================
# Utility Functions
# ============================================================================

def create_weighted_reward(
    reward_functions: Dict[str, tuple[Callable, float]]
) -> Callable:
    """
    Create a weighted combination of multiple reward functions.

    Example:
        combined_reward = create_weighted_reward({
            "accuracy": (accuracy_reward, 0.5),
            "speed": (speed_reward, 0.3),
            "quality": (quality_reward, 0.2)
        })

    Args:
        reward_functions: Dict mapping names to (function, weight) tuples

    Returns:
        Combined reward function
    """
    async def combined_reward(task: Dict, result: Dict) -> float:
        total_reward = 0.0
        total_weight = sum(weight for _, weight in reward_functions.values())

        for name, (func, weight) in reward_functions.items():
            try:
                reward = func(task, result)
                if asyncio.iscoroutine(reward):
                    reward = await reward
                total_reward += reward * weight
            except Exception as e:
                log.error(f"Error in reward function {name}: {str(e)}")

        return total_reward / total_weight if total_weight > 0 else 0.0

    return combined_reward
