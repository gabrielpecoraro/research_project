import torch
from pathfinding import create_sample_neighborhood, AStar
from pursuit_environment import PursuitEnvironment


def test_expert_system():
    """Test the expert pathfinding system"""
    print("Testing expert pathfinding system...")

    try:
        from expert_pathfinding_system import ExpertPathfindingSystem

        # Create a small environment for testing
        base_env = create_sample_neighborhood(width=20, height=20)
        env = PursuitEnvironment(20, 20)
        env.env = base_env

        expert_system = ExpertPathfindingSystem(env)

        # Test action generation
        agent_pos = (1.0, 1.0)
        target_pos = (15.0, 15.0)

        action = expert_system.generate_expert_action(agent_pos, target_pos)
        print(f"Expert action from {agent_pos} to {target_pos}: {action}")

        # Test coordination
        agent_positions = [(1.0, 1.0), (2.0, 1.0)]
        actions = expert_system.generate_coordination_actions(
            agent_positions, target_pos
        )
        print(f"Coordinated actions: {actions}")

        print("Expert system test: PASSED")
        return True

    except Exception as e:
        print(f"Expert system test: FAILED - {e}")
        return False


def test_training_integration():
    """Test a short training run"""
    print("Testing pathfinding-guided training...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        from train_qmix import QMixTrainer

        trainer = QMixTrainer(
            device=device, use_pathfinding_guidance=True, guidance_weight=0.2
        )

        # Run just a few episodes to test
        trained_agent = trainer.train(
            num_episodes=5, visualize_every=100, save_every=50, expert_ratio=0.5
        )

        print("Training integration test: PASSED")
        return True

    except Exception as e:
        print(f"Training integration test: FAILED - {e}")
        return False


if __name__ == "__main__":
    print("=== TESTING PATHFINDING INTEGRATION ===")

    success = True
    success &= test_expert_system()
    success &= test_training_integration()

    if success:
        print("\n✅ All tests passed! Integration is working correctly.")
        print("You can now run the full training with: python train_qmix.py")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
