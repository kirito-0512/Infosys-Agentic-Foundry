# © 2024-25 Infosys Limited, Bangalore, India. All Rights Reserved.
"""
Example: Train a React Agent using APO (Automatic Prompt Optimization).

This script demonstrates how to train an existing Foundry React Agent
using Agent Lightning's APO algorithm for prompt optimization.

APO uses beam search and LLM-generated critiques to improve prompts
without requiring gradients or GPU training.

Usage:
    python train_react_agent_apo.py --agent-id <agent_id> --dataset-file <path>
"""

import asyncio
import argparse
import json
from pathlib import Path

from agentlightning import Trainer
from agentlightning.algorithm.apo import APO

# Import Foundry components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.dependencies import ServiceProvider
from poc_agent_lightning.src.foundry_lit_agent_adapter import ReactAgentAdapter
from poc_agent_lightning.src.reward_functions import RewardFunctionRegistry


async def load_training_dataset(dataset_file: Path) -> list[dict]:
    """
    Load training dataset from JSON file.

    Expected format:
    [
        {
            "query": "Find all customers in California",
            "expected_sql": "SELECT * FROM customers WHERE state = 'CA'",
            "db_schema": {...}
        },
        ...
    ]
    """
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} training examples")
    return dataset


async def main():
    parser = argparse.ArgumentParser(description="Train React Agent with APO")
    parser.add_argument("--agent-id", required=True, help="Foundry agent ID")
    parser.add_argument("--dataset-file", required=True, help="Path to training dataset JSON")
    parser.add_argument("--reward-function", default="sql_correctness",
                        help="Reward function name from registry")
    parser.add_argument("--n-iterations", type=int, default=5,
                        help="Number of APO iterations")
    parser.add_argument("--beam-width", type=int, default=3,
                        help="Beam search width")
    parser.add_argument("--n-rollouts-per-prompt", type=int, default=10,
                        help="Rollouts per prompt candidate")
    parser.add_argument("--n-runners", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--output-dir", default="./apo_output",
                        help="Output directory for checkpoints")

    args = parser.parse_args()

    print("=" * 80)
    print("React Agent Training with APO (Automatic Prompt Optimization)")
    print("=" * 80)
    print(f"Agent ID: {args.agent_id}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Reward Function: {args.reward_function}")
    print(f"APO Iterations: {args.n_iterations}")
    print(f"Beam Width: {args.beam_width}")
    print("=" * 80)

    # 1. Load training dataset
    print("\n[1/5] Loading training dataset...")
    train_dataset = await load_training_dataset(Path(args.dataset_file))

    # Optional: Split into train/val
    split_idx = int(len(train_dataset) * 0.8)
    train_data = train_dataset[:split_idx]
    val_data = train_dataset[split_idx:]
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val: {len(val_data)} examples")

    # 2. Get Foundry services
    print("\n[2/5] Initializing Foundry services...")
    agent_service = ServiceProvider.get_agent_service()

    # Verify agent exists
    agent_config = await agent_service.get_agent(args.agent_id)
    if not agent_config:
        raise ValueError(f"Agent {args.agent_id} not found")

    agent_name = agent_config[0]["agentic_application_name"]
    print(f"  Agent Name: {agent_name}")
    print(f"  Agent Type: {agent_config[0]['agentic_application_type']}")

    # 3. Create LitAgent adapter
    print("\n[3/5] Creating LitAgent adapter...")
    reward_function = RewardFunctionRegistry.get(args.reward_function)
    if not reward_function:
        raise ValueError(f"Reward function '{args.reward_function}' not found")

    lit_agent = ReactAgentAdapter(
        agent_id=args.agent_id,
        agent_service=agent_service,
        reward_function=reward_function
    )
    print(f"  Adapter created for agent: {agent_name}")

    # 4. Configure APO algorithm
    print("\n[4/5] Configuring APO algorithm...")
    apo_config = {
        "n_iterations": args.n_iterations,
        "beam_width": args.beam_width,
        "n_rollouts_per_prompt": args.n_rollouts_per_prompt,
        "output_dir": args.output_dir,
    }
    print(f"  APO Config: {json.dumps(apo_config, indent=2)}")

    # 5. Create trainer and run
    print("\n[5/5] Starting APO training...")
    trainer = Trainer(
        algorithm=APO(apo_config),
        n_runners=args.n_runners,
    )

    print("\nTraining in progress...")
    print("This may take several hours depending on dataset size and beam width.")
    print("=" * 80)

    # Run training
    results = await trainer.fit(lit_agent, train_data, val_data)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Print results
    print("\nTraining Results:")
    print(f"  Best Prompt Found: {results.get('best_resources', {}).get('system_prompt', 'N/A')[:200]}...")
    print(f"  Improvement: {results.get('improvement_pct', 0):.1f}%")
    print(f"  Final Score: {results.get('final_score', 0):.3f}")
    print(f"  Baseline Score: {results.get('baseline_score', 0):.3f}")

    # 6. Update agent with best prompt
    print("\nUpdating agent with optimized prompt...")
    best_prompt = results["best_resources"]["system_prompt"]

    await agent_service.update_agent(
        agentic_application_id=args.agent_id,
        system_prompt={"SYSTEM_PROMPT_REACT_AGENT": best_prompt}
    )

    print(f"\n✅ Agent {agent_name} updated successfully!")
    print(f"   Checkpoints saved to: {args.output_dir}")
    print(f"   Ready for production deployment.")


if __name__ == "__main__":
    asyncio.run(main())
