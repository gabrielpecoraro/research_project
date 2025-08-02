import matplotlib.pyplot as plt
import numpy as np
from pursuit_environment import create_test_environments


def visualize_environment(env, title="Urban Environment"):
    """Visualize a single environment"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot obstacles (buildings)
    for ox, oy, ow, oh in env.env.obstacles:
        rect = plt.Rectangle(
            (ox, oy), ow, oh, color="gray", alpha=0.8, edgecolor="black"
        )
        ax.add_patch(rect)

    # Plot agents
    colors = ["red", "blue"]
    for i, pos in enumerate(env.agent_positions):
        ax.plot(
            pos[0],
            pos[1],
            "o",
            color=colors[i],
            markersize=12,
            label=f"Agent {i + 1}",
            markeredgecolor="white",
            markeredgewidth=2,
        )

    # Plot target
    ax.plot(
        env.target.position[0],
        env.target.position[1],
        "s",
        color="green",
        markersize=12,
        label="Target",
        markeredgecolor="white",
        markeredgewidth=2,
    )

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    return fig, ax


def visualize_all_environments():
    """Visualize all test environments"""
    environments = create_test_environments()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, env in enumerate(environments):
        if i < len(axes):
            ax = axes[i]

            # Reset environment to get random positions
            env.reset()

            # Plot obstacles (buildings)
            for ox, oy, ow, oh in env.env.obstacles:
                rect = plt.Rectangle(
                    (ox, oy), ow, oh, color="gray", alpha=0.8, edgecolor="black"
                )
                ax.add_patch(rect)

            # Plot agents
            colors = ["red", "blue"]
            for j, pos in enumerate(env.agent_positions):
                ax.plot(
                    pos[0],
                    pos[1],
                    "o",
                    color=colors[j],
                    markersize=10,
                    markeredgecolor="white",
                    markeredgewidth=1,
                )

            # Plot target
            ax.plot(
                env.target.position[0],
                env.target.position[1],
                "s",
                color="green",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1,
            )

            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_title(f"Urban Environment {env.width}x{env.height}")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

    # Hide unused subplots
    for i in range(len(environments), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle(
        "Generated Urban Environments with Random Spawn Positions", fontsize=16, y=0.98
    )
    plt.show()


if __name__ == "__main__":
    visualize_all_environments()
