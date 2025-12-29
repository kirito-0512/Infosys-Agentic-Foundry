# © 2024-25 Infosys Limited, Bangalore, India. All Rights Reserved.
"""
Example: Train a React Agent using VERL (Reinforcement Learning with PPO).

This script demonstrates how to train an existing Foundry React Agent
using Agent Lightning's VERL algorithm with PPO (Proximal Policy Optimization).

VERL requires GPU training and is optimized for the 4x A100 80GB setup.

Usage:
    python train_react_agent_verl.py --agent-id <agent_id> --dataset-file <path>
"""

import asyncio
import argparse
import json
from pathlib import Path

from agentlightning import Trainer
from agentlightning.algorithm.verl import VERL

# Import Foundry components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.dependencies import ServiceProvider
from poc_agent_lightning.src.foundry_lit_agent_adapter import ReactAgentAdapter
from poc_agent_lightning.src.reward_functions import RewardFunctionRegistry


async def load_training_dataset(dataset_file: Path) -> list[dict]:
    """Load training dataset from JSON file."""
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} training examples")
    return dataset


async def main():
    parser = argparse.ArgumentParser(description="Train React Agent with VERL/PPO")
    parser.add_argument("--agent-id", required=True, help="Foundry agent ID")
    parser.add_argument("--dataset-file", required=True, help="Path to training dataset JSON")
    parser.add_argument("--reward-function", default="sql_correctness",
                        help="Reward function name")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--total-epochs", type=int, default=10,
                        help="Total training epochs")
    parser.add_argument("--train-batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--n-runners", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--use-trajectory-agg", action="store_true",
                        help="Enable trajectory-level aggregation (experimental)")
    parser.add_argument("--output-dir", default="./verl_output",
                        help="Output directory for checkpoints")

    args = parser.parse_args()

    print("=" * 80)
    print("React Agent Training with VERL (PPO Reinforcement Learning)")
    print("=" * 80)
    print(f"Agent ID: {args.agent_id}")
    print(f"Base Model: {args.base_model}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Reward Function: {args.reward_function}")
    print(f"Total Epochs: {args.total_epochs}")
    print(f"Batch Size: {args.train_batch_size}")
    print(f"GPU: 4x A100 80GB")
    print("=" * 80)

    # 1. Load training dataset
    print("\n[1/6] Loading training dataset...")
    train_dataset = await load_training_dataset(Path(args.dataset_file))

    # Split into train/val
    split_idx = int(len(train_dataset) * 0.8)
    train_data = train_dataset[:split_idx]
    val_data = train_dataset[split_idx:]
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val: {len(val_data)} examples")

    # 2. Get Foundry services
    print("\n[2/6] Initializing Foundry services...")
    agent_service = ServiceProvider.get_agent_service()

    agent_config = await agent_service.get_agent(args.agent_id)
    if not agent_config:
        raise ValueError(f"Agent {args.agent_id} not found")

    agent_name = agent_config[0]["agentic_application_name"]
    print(f"  Agent Name: {agent_name}")

    # 3. Create LitAgent adapter
    print("\n[3/6] Creating LitAgent adapter...")
    reward_function = RewardFunctionRegistry.get(args.reward_function)
    if not reward_function:
        raise ValueError(f"Reward function '{args.reward_function}' not found")

    lit_agent = ReactAgentAdapter(
        agent_id=args.agent_id,
        agent_service=agent_service,
        reward_function=reward_function
    )

    # 4. Configure VERL for 4x A100 80GB
    print("\n[4/6] Configuring VERL for 4x A100...")
    verl_config = {
        # Algorithm configuration
        "algorithm": {
            "adv_estimator": "grpo",  # Group Relative Policy Optimization
            "use_kl_in_reward": False,
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": 0.001,
            },
        },

        # Data configuration
        "data": {
            "train_batch_size": args.train_batch_size,  # Total batch size across GPUs
            "val_batch_size": 32,
            "max_prompt_length": 4096,
            "max_response_length": 2048,
            "shuffle": True,
        },

        # Model configuration (Actor, Reference, Critic)
        "actor_rollout_ref": {
            "model": {
                "path": args.base_model,
                "trust_remote_code": True,
            },

            # vLLM configuration for fast inference
            "rollout": {
                "name": "vllm",
                "gpu_memory_utilization": 0.4,  # Leave room for training
                "tensor_parallel_size": 2,  # Use 2 GPUs for vLLM
            },

            # PPO actor training configuration
            "actor": {
                "optim": {
                    "lr": 1e-6,
                    "weight_decay": 0.01,
                },
                "ppo_mini_batch_size": 8,  # Per-GPU mini-batch
                "ppo_epochs": 2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "entropy_coeff": 0.0,
            },

            # Reference model (frozen)
            "ref": {
                "log_prob_micro_batch_size": 8,
            },
        },

        # Critic model configuration
        "critic": {
            "optim": {
                "lr": 1e-5,
                "weight_decay": 0.01,
            },
            "model": {
                "path": args.base_model,  # Can use same or different model
            },
            "ppo_micro_batch_size": 8,
        },

        # Trainer configuration
        "trainer": {
            "total_epochs": args.total_epochs,
            "save_freq": 1,  # Save checkpoint every epoch
            "test_freq": 1,  # Validate every epoch
            "project_name": f"verl-{args.agent_id}",
            "experiment_name": f"training-{args.agent_id}",
            "logger": ["console", "wandb"],  # W&B for visualization
            "default_hdfs_dir": args.output_dir,
        },

        # FSDP (Fully Sharded Data Parallel) for 4x A100
        "trainer_fsdp": {
            "use_fsdp": True,
            "sharding_strategy": "FULL_SHARD",  # Maximum memory efficiency
            "backward_prefetch": "BACKWARD_PRE",
            "forward_prefetch": True,
            "limit_all_gathers": True,
        },

        # Agent Lightning specific
        "agentlightning": {
            "store": {
                "type": "mongo",
                "uri": "mongodb://localhost:27017",
                "db_name": f"training_{args.agent_id}",
            },
        },
    }

    # Optional: Enable trajectory-level aggregation
    if args.use_trajectory_agg:
        verl_config["agentlightning"]["trace_aggregator"] = {
            "level": "trajectory",
            "trajectory_max_prompt_length": 4096,
            "trajectory_max_response_length": 34384,
        }
        print("  Trajectory aggregation: ENABLED (experimental)")

    print("  VERL configuration optimized for 4x A100 80GB")
    print(f"  Model: {args.base_model}")
    print(f"  Batch size: {args.train_batch_size} (distributed)")
    print(f"  FSDP: Enabled (Full Shard)")

    # 5. Create trainer
    print("\n[5/6] Creating VERL trainer...")
    trainer = Trainer(
        algorithm=VERL(verl_config),
        n_runners=args.n_runners,  # Parallel workers for rollouts
    )

    print(f"  {args.n_runners} parallel workers for rollout generation")
    print("  Training on 4 GPUs with FSDP")

    # 6. Start training
    print("\n[6/6] Starting VERL training...")
    print("\nThis will take several hours (estimated: 8-12 hours)")
    print("Monitor progress at: http://localhost:6006 (TensorBoard)")
    print("Or Weights & Biases dashboard if configured")
    print("=" * 80)

    # Run training
    results = await trainer.fit(lit_agent, train_data, val_data)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Print results
    print("\nTraining Results:")
    print(f"  Final Reward: {results.get('final_reward', 0):.3f}")
    print(f"  Baseline Reward: {results.get('baseline_reward', 0):.3f}")
    print(f"  Improvement: {results.get('improvement_pct', 0):.1f}%")
    print(f"  Best Epoch: {results.get('best_epoch', 'N/A')}")

    # 7. Deploy best checkpoint
    print("\nDeploying best checkpoint to production...")

    # In production, you would:
    # 1. Load best model checkpoint
    # 2. Run validation suite
    # 3. A/B test against baseline
    # 4. Gradual rollout

    print(f"\n✅ Training complete for agent {agent_name}")
    print(f"   Checkpoints saved to: {args.output_dir}")
    print(f"   Best model available for deployment")

    print("\nNext steps:")
    print("  1. Review checkpoints and select best epoch")
    print("  2. Run comprehensive evaluation on held-out test set")
    print("  3. A/B test trained model vs. baseline")
    print("  4. Deploy to production if metrics improve")


if __name__ == "__main__":
    asyncio.run(main())
