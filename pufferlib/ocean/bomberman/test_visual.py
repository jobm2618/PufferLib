#!/usr/bin/env python3
"""Visual test for Bomberman environment

Run this to see the environment rendering in action.
Press ESC to exit.

Usage:
    python test_visual.py
"""

from bomberman import Bomberman
import numpy as np

def main():
    print("=== Bomberman Visual Test ===")
    print("Controls: Press ESC to exit")
    print("Starting environment...\n")

    # Create environment with rendering enabled
    env = Bomberman(
        num_envs=1,
        num_agents=4,
        width=21,
        height=21,
        vision=5,
        bomb_timer=30,
        render_mode='human'
    )

    obs, _ = env.reset()

    print("Environment running. Watch the window!")
    print("Agents are colored circles.")
    print("Random actions being taken...\n")

    # Run for a while with random actions
    step = 0
    while True:
        # Random actions (more likely to place bombs for interesting gameplay)
        actions = np.random.randint(0, 9, size=4)
        # Bias toward bomb placement actions (5-8)
        for i in range(4):
            if np.random.random() < 0.3:  # 30% chance
                actions[i] = np.random.randint(5, 9)

        obs, rewards, terminals, truncations, info = env.step(actions)
        env.render()

        step += 1

        # Log interesting events
        if np.any(rewards > 0.1):
            print(f"Step {step}: Big rewards! {rewards}")
        if np.any(terminals):
            print(f"Step {step}: Agent(s) died! Terminals: {terminals}")

            # Reset after all agents die
            if np.all(terminals):
                print("\nAll agents dead! Resetting...\n")
                env.reset()
                step = 0

    env.close()


if __name__ == '__main__':
    main()
