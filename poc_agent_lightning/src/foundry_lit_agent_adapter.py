# © 2024-25 Infosys Limited, Bangalore, India. All Rights Reserved.
"""
FoundryLitAgentAdapter: Wrapper to integrate Infosys Agentic Foundry agents with Agent Lightning.

This adapter enables any Foundry agent to be trained using Agent Lightning's
reinforcement learning infrastructure with minimal code changes.
"""

from typing import Callable, Dict, Any, Optional
from agentlightning import LitAgent, emit_reward, emit_annotation
from agentlightning.types import Rollout, NamedResources

from src.database.services import AgentService
from src.inference.base_agent_inference import BaseAgentInference
from src.api.dependencies import ServiceProvider
from telemetry_wrapper import logger as log


class FoundryLitAgentAdapter(LitAgent[Dict[str, Any]]):
    """
    Adapts Infosys Agentic Foundry agents to Agent Lightning's LitAgent interface.

    This adapter wraps a Foundry agent's inference engine and enables RL training
    by handling reward emission, resource updates, and trace collection.

    Args:
        agent_id: Unique identifier of the Foundry agent
        agent_service: Foundry's AgentService for retrieving agent configuration
        inference_service: Foundry's inference engine for the agent type
        reward_function: Function that computes rewards from task and result
        enable_resource_updates: Whether to update agent resources from training

    Example:
        >>> adapter = FoundryLitAgentAdapter(
        ...     agent_id="react-agent-sql-123",
        ...     agent_service=agent_service,
        ...     inference_service=react_agent_inference,
        ...     reward_function=sql_correctness_reward
        ... )
        >>> trainer = Trainer(algorithm=VERL(config))
        >>> trainer.fit(adapter, train_dataset)
    """

    def __init__(
        self,
        agent_id: str,
        agent_service: AgentService,
        inference_service: BaseAgentInference,
        reward_function: Callable[[Dict, Dict], float],
        enable_resource_updates: bool = True,
        debug_mode: bool = False
    ):
        super().__init__()
        self.agent_id = agent_id
        self.agent_service = agent_service
        self.inference_service = inference_service
        self.reward_function = reward_function
        self.enable_resource_updates = enable_resource_updates
        self.debug_mode = debug_mode

        # Cache agent configuration
        self._agent_config_cache: Optional[Dict] = None

    async def _get_agent_config(self) -> Dict:
        """Retrieve agent configuration from Foundry database."""
        if self._agent_config_cache is None:
            log.info(f"Loading agent configuration for {self.agent_id}")
            agent_data = await self.agent_service.get_agent(
                agentic_application_id=self.agent_id
            )
            if not agent_data:
                raise ValueError(f"Agent {self.agent_id} not found")
            self._agent_config_cache = agent_data[0]

        return self._agent_config_cache

    async def _update_agent_config_from_resources(
        self,
        agent_config: Dict,
        resources: NamedResources
    ) -> Dict:
        """
        Update agent configuration with resources from Lightning Store.

        Resources can include:
        - system_prompt: Updated system prompts (for prompt optimization)
        - model_weights: Fine-tuned model weights (for VERL training)
        - hyperparameters: Agent-specific parameters

        Args:
            agent_config: Current agent configuration
            resources: Resources from Lightning Store (training-optimized)

        Returns:
            Updated agent configuration
        """
        updated_config = agent_config.copy()

        # Update system prompts if provided
        if "system_prompt" in resources:
            system_prompt_updates = resources["system_prompt"]

            if isinstance(system_prompt_updates, dict):
                # Multi-Agent: Multiple system prompts (planner, executor, etc.)
                for key, value in system_prompt_updates.items():
                    if key in updated_config.get("system_prompt", {}):
                        updated_config["system_prompt"][key] = value
                        if self.debug_mode:
                            log.debug(f"Updated {key} from resources")
            elif isinstance(system_prompt_updates, str):
                # Single-Agent: One system prompt
                prompt_key = list(updated_config.get("system_prompt", {}).keys())[0]
                updated_config["system_prompt"][prompt_key] = system_prompt_updates
                if self.debug_mode:
                    log.debug(f"Updated system prompt from resources")

        # Update model configuration if provided
        if "model_config" in resources:
            updated_config["model_name"] = resources["model_config"].get(
                "model_name",
                updated_config.get("model_name")
            )

        # Update hyperparameters if provided
        if "hyperparameters" in resources:
            updated_config.setdefault("hyperparameters", {})
            updated_config["hyperparameters"].update(resources["hyperparameters"])

        return updated_config

    async def rollout(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a single training rollout.

        This method is called by Agent Lightning runners during training.
        It executes the Foundry agent, computes rewards, and emits them
        for the training algorithm.

        Args:
            task: Training task containing:
                - query: User query to process
                - context: Optional context information
                - expected_output: Optional expected output for evaluation
            resources: Resources from Lightning Store (e.g., optimized prompts)
            rollout: Rollout metadata from Lightning

        Returns:
            Agent execution result (or None if tracer handles everything)
        """
        try:
            # 1. Get agent configuration
            agent_config = await self._get_agent_config()

            # 2. Update configuration with resources from training
            if self.enable_resource_updates and resources:
                agent_config = await self._update_agent_config_from_resources(
                    agent_config,
                    resources
                )

            # 3. Extract task details
            user_query = task.get("query", "")
            context = task.get("context", {})
            session_id = rollout.rollout_id  # Use rollout ID as session

            if self.debug_mode:
                emit_annotation("task_query", user_query)
                emit_annotation("rollout_mode", rollout.mode)

            # 4. Execute agent using Foundry's inference engine
            # The Lightning tracer automatically captures all LLM calls, tool usage, etc.
            log.info(f"Executing agent {self.agent_id} for rollout {rollout.rollout_id}")

            result = await self.inference_service.process(
                agent_config=agent_config,
                user_query=user_query,
                session_id=session_id,
                **context  # Pass any additional context
            )

            if self.debug_mode:
                emit_annotation("result_status", result.get("status", "unknown"))

            # 5. Compute reward
            try:
                reward = self.reward_function(task, result)

                # Handle both scalar and dict rewards
                if isinstance(reward, dict):
                    # Multi-dimensional reward
                    emit_reward(reward, primary_key=reward.get("_primary", "total"))
                    if self.debug_mode:
                        for key, value in reward.items():
                            emit_annotation(f"reward_{key}", value)
                else:
                    # Single-dimensional reward
                    emit_reward(reward)
                    if self.debug_mode:
                        emit_annotation("reward", reward)

                log.info(f"Emitted reward for rollout {rollout.rollout_id}: {reward}")

            except Exception as e:
                log.error(f"Error computing reward: {str(e)}")
                # Emit zero reward on error (don't fail the rollout)
                emit_reward(0.0)
                emit_annotation("reward_error", str(e))

            # 6. Return result (tracer captures everything automatically)
            return result

        except Exception as e:
            log.error(f"Error in rollout {rollout.rollout_id}: {str(e)}")
            emit_reward(0.0)  # Zero reward for failed rollouts
            emit_annotation("error", str(e))
            raise


class ReactAgentAdapter(FoundryLitAgentAdapter):
    """Specialized adapter for React Agent template."""

    def __init__(self, agent_id: str, agent_service: AgentService, reward_function: Callable):
        inference_service = ServiceProvider.get_specialized_inference_service("react_agent")
        super().__init__(
            agent_id=agent_id,
            agent_service=agent_service,
            inference_service=inference_service,
            reward_function=reward_function
        )


class MultiAgentAdapter(FoundryLitAgentAdapter):
    """
    Specialized adapter for Multi-Agent template.

    Handles multiple system prompts (planner, executor, critic, etc.)
    and supports multi-objective reward functions.
    """

    def __init__(self, agent_id: str, agent_service: AgentService, reward_function: Callable):
        inference_service = ServiceProvider.get_specialized_inference_service("multi_agent")
        super().__init__(
            agent_id=agent_id,
            agent_service=agent_service,
            inference_service=inference_service,
            reward_function=reward_function
        )

    async def _update_agent_config_from_resources(
        self,
        agent_config: Dict,
        resources: NamedResources
    ) -> Dict:
        """Override to handle multi-agent system prompts."""
        updated_config = await super()._update_agent_config_from_resources(
            agent_config,
            resources
        )

        # Multi-agent specific: Update all sub-agent prompts
        if "system_prompt" in resources and isinstance(resources["system_prompt"], dict):
            for prompt_key in [
                "SYSTEM_PROMPT_PLANNER_AGENT",
                "SYSTEM_PROMPT_EXECUTOR_AGENT",
                "SYSTEM_PROMPT_CRITIC_AGENT",
                "SYSTEM_PROMPT_RESPONSE_GENERATOR_AGENT",
                "SYSTEM_PROMPT_GENERAL_LLM"
            ]:
                if prompt_key in resources["system_prompt"]:
                    updated_config["system_prompt"][prompt_key] = resources["system_prompt"][prompt_key]

        return updated_config


class MetaAgentAdapter(FoundryLitAgentAdapter):
    """
    Specialized adapter for Meta-Agent template.

    Supports hierarchical RL where meta-agent learns to coordinate worker agents.
    """

    def __init__(self, agent_id: str, agent_service: AgentService, reward_function: Callable):
        inference_service = ServiceProvider.get_specialized_inference_service("meta_agent")
        super().__init__(
            agent_id=agent_id,
            agent_service=agent_service,
            inference_service=inference_service,
            reward_function=reward_function
        )

    async def rollout(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout
    ) -> Optional[Dict[str, Any]]:
        """
        Execute meta-agent rollout with hierarchical reward tracking.

        Meta-agents delegate to worker agents, so we track:
        - Meta-agent decision quality
        - Worker agent performance
        - Overall task completion
        """
        # Execute base rollout
        result = await super().rollout(task, resources, rollout)

        # Emit hierarchical annotations
        if result and "worker_results" in result:
            emit_annotation("num_workers_used", len(result["worker_results"]))
            for idx, worker_result in enumerate(result["worker_results"]):
                emit_annotation(f"worker_{idx}_status", worker_result.get("status"))

        return result
