'''
Multi-agent battle royale style Bomberman environment supporting 2-64 agents per environment.
'''

import math
import numpy as np
import gymnasium

import pufferlib
from pufferlib import APIUsageError
from pufferlib.ocean.bomberman import binding


def auto_grid_size(num_agents, cells_per_agent=64):
    """Calculate appropriate grid size for agent count with hard limits.

    Args:
        num_agents: Number of agents in the environment
        cells_per_agent: Target cells per agent (fixed at 64)

    Returns:
        (width, height) tuple of odd integers for symmetry

    Raises:
        ValueError: If num_agents > 64 or calculated grid > 101×101
    """
    # Enforce maximum agents per environment
    if num_agents > 64:
        raise ValueError(f"Too many agents: {num_agents} (max 64 per environment)")

    total_cells = num_agents * cells_per_agent
    side = int(math.sqrt(total_cells))

    # Force odd for symmetry (center spawn, etc.)
    if side % 2 == 0:
        side += 1

    # Enforce minimum (at least 11×11 for vision=5)
    side = max(side, 11)

    # Enforce maximum (101×101 = 10,201 cells, fits in L2 cache)
    if side > 101:
        raise ValueError(
            f"Grid too large: {side}×{side} for {num_agents} agents "
            f"(max 101×101, consider splitting into more environments)"
        )

    return side, side


class Bomberman(pufferlib.PufferEnv):
    """Multi-agent Bomberman environment.

    Features:
    - Scalable: 2-64 agents per environment
    - Auto-sizing: Grid scales with agent count
    - Local vision: 11×11 observation space per agent
    
    Game mechanics:
    - Movement in 4 directions
    - Bomb placement with timed explosions
    - Destructible blocks
    - Agent elimination on fire contact

    Observation space (per agent):
        Box(low=0, high=1, shape=(972,), dtype=float32)
        Flattened concatenation of:
          Spatial (11×11×8 = 968 values):
            0: Walls (hard, indestructible)
            1: Soft blocks (destructible)
            2: Other agents (normalized by num_agents)
            3: Bombs (timer normalized to [0,1])
            4: Fires (1.0 = active fire)
            5: Bomb power-ups (1.0 = present)
            6: Blast power-ups (1.0 = present)
            7: Speed power-ups (1.0 = present)
          Scalar (4 values):
            0: Remaining lives (0.33, 0.67, or 1.0 for 1/2/3 lives)
            1: Bomb count (current/max)
            2: Blast radius ((current-2)/(max-2))
            3: Speed multiplier ((speed-1.0)/(max-1.0))

    Action space (per agent):
        Discrete(9):
          0: NOOP
          1-4: UP, DOWN, LEFT, RIGHT (move only)
          5-8: UP+BOMB, DOWN+BOMB, LEFT+BOMB, RIGHT+BOMB

    Rewards:
        Placement: +(num_agents-rank)/num_agents when eliminated (linear)
        Survival: +(agent_steps/game_ticks) when eliminated

    Lives system:
        Agents have 3 lives (take 3 hits before elimination)
        Fire contact: -1 life (not instant death)
        At 0 lives: terminal state, placement reward given
    """

    def __init__(self, num_envs=1, num_agents=8,
                 width=None, height=None,
                 max_ticks=10000,
                 render_mode=None,
                 log_interval=128,
                 buf=None,
                 seed=0):
        """Initialize Bomberman environment.

        Fixed parameters (not configurable):
        - vision=5 (11×11 window)
        - cells_per_agent=64 (auto-sizing)
        - bomb_timer=5 (0.5 seconds at 10 FPS)
        - explosion_duration=3 (0.3 seconds)
        - blast_radius=2 (baseline, up to 5 with power-ups)
        - block_density=0.4 (40% of interior)
        - starting_lives=3 (per agent)

        Args:
            num_envs: Number of parallel environments
            num_agents: Agents per environment (2-64)
            width: Grid width (auto-calculated if None)
            height: Grid height (auto-calculated if None)
            max_ticks: Maximum ticks per episode (for battle royale)
            render_mode: 'human' for visualization, None for no render
            log_interval: Ticks between logging metrics
            buf: PufferLib buffer (internal use)
            seed: Random seed
        """
        # Fixed parameters
        cells_per_agent = 64
        vision = 5
        bomb_timer = 5
        explosion_duration = 3
        blast_radius = 2
        block_density = 0.4

        # Auto-size grid if not specified
        if width is None or height is None:
            width, height = auto_grid_size(num_agents, cells_per_agent)
            if num_envs == 1:  # Only print for single env to avoid spam
                print(f"Auto-sized grid: {width}×{height} for {num_agents} agents")

        # Validate parameters
        if num_agents > 64:
            raise APIUsageError(f"Too many agents: {num_agents} (max 64)")

        if width > 101 or height > 101:
            raise APIUsageError(f"Grid too large: {width}×{height} (max 101×101)")

        min_cells = num_agents * cells_per_agent
        if width * height < min_cells:
            raise APIUsageError(
                f"Grid too small: {width}×{height}={width*height} cells "
                f"for {num_agents} agents (need {min_cells}+ cells)"
            )

        min_size = 2 * vision + 3  # Vision + buffer for walls
        if width < min_size or height < min_size:
            raise APIUsageError(
                f"Grid too small for vision={vision}: need at least {min_size}×{min_size}"
            )

        # Define observation space: flattened spatial + scalars
        window = 2 * vision + 1  # 11
        num_spatial_channels = 8  # walls, blocks, others, bombs, fires, 3 power-ups
        spatial_size = window * window * num_spatial_channels  # 11×11×8 = 968
        scalar_size = 4  # lives, bomb_count, blast_radius, speed
        total_obs_size = spatial_size + scalar_size  # 972

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1,
            shape=(total_obs_size,),
            dtype=np.float32
        )

        # Define action space: 9 discrete actions
        self.single_action_space = gymnasium.spaces.Discrete(9)

        # Multi-agent configuration
        self.num_agents = num_envs * num_agents
        self.render_mode = render_mode
        self.log_interval = log_interval

        # Python tick: Tracks global steps across ALL vectorized environments
        # Used for logging intervals (e.g., log every 128 global steps)
        # C envs each maintain their own tick for per-env game state
        self.tick = 0

        # Store configuration for later reference
        self.width = width
        self.height = height
        self.vision = vision
        self.agents_per_env = num_agents

        # Initialize PufferEnv (creates shared buffers)
        super().__init__(buf)

        # Create C environments with buffer slicing
        c_envs = []
        for i in range(num_envs):
            # Slice buffers for this environment
            offset_start = i * num_agents
            offset_end = (i + 1) * num_agents

            obs_slice = self.observations[offset_start:offset_end]
            act_slice = self.actions[offset_start:offset_end]
            rew_slice = self.rewards[offset_start:offset_end]
            term_slice = self.terminals[offset_start:offset_end]
            trunc_slice = self.truncations[offset_start:offset_end]

            # Unique seed per environment
            env_seed = i + seed * num_envs

            # Initialize C environment
            env_id = binding.env_init(
                obs_slice,
                act_slice,
                rew_slice,
                term_slice,
                trunc_slice,
                env_seed,
                width=width,
                height=height,
                num_agents=num_agents,
                vision=vision,
                bomb_timer=bomb_timer,
                explosion_duration=explosion_duration,
                blast_radius=blast_radius,
                block_density=block_density,
                max_ticks=max_ticks
            )
            c_envs.append(env_id)

        # Vectorize all environments
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        """Reset all environments to initial state.

        Returns:
            observations: (num_agents, 972) array (flattened spatial + scalars)
            info: List of info dicts
        """
        # Reset global tick counter (Python-side tracking)
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        """Execute one step in all environments.

        Args:
            actions: (num_agents,) array of action indices [0-8]

        Returns:
            observations: (num_agents, 972) array (flattened spatial + scalars)
            rewards: (num_agents,) array of float rewards
            terminals: (num_agents,) array of bool episode ends
            truncations: (num_agents,) array of bool truncations
            info: List of info dicts (metrics logged every log_interval ticks)
        """
        self.actions[:] = actions

        # Increment global tick (Python-side tracking for logging)
        # C envs each maintain their own tick for per-env game state
        self.tick += 1

        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log_dict = binding.vec_log(self.c_envs)
            if log_dict:  # May be empty if no episodes ended
                info.append(log_dict)

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        """Render the first environment (env_id=0).

        Only renders env 0 to avoid opening multiple windows.
        Press ESC to exit visualization.
        """
        binding.vec_render(self.c_envs, 0)

    def close(self):
        """Clean up resources and close environments."""
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024):
    """Performance benchmark for training pipeline (Python + C integration).

    Tests the full Python API with vectorized environments to measure
    realistic training performance including Python overhead.
    For C-only benchmark, use bomberman.c test_performance().

    Args:
        timeout: Run for this many seconds
        atn_cache: Number of random actions to pregenerate
    """
    print("Initializing Bomberman environment...")
    env = Bomberman(num_envs=64, num_agents=8)
    env.reset()
    tick = 0

    total_agents = env.num_agents
    actions = np.random.randint(0, 9, (atn_cache, total_agents))

    print(f"Running benchmark for {timeout} seconds...")
    print(f"Config: {64} envs × {8} agents = {total_agents} agents")

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    elapsed = time.time() - start
    sps = total_agents * tick / elapsed

    print(f"\n=== Bomberman Performance ===")
    print(f"Total steps: {tick:,}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"SPS: {sps:,.0f} steps/second")

if __name__ == '__main__':
    test_performance()
