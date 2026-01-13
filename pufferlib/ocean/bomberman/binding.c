/* Bomberman Python-C binding
 *
 * Uses PufferLib's env_binding.h template for zero-copy buffer sharing
 * This file bridges the Python API (bomberman.py) with C implementation (bomberman.c)
 */

#include "bomberman.h"

// Define Env type for env_binding.h template
#define Env Bomberman

// Include PufferLib's binding template
// This provides: env_init, env_reset, env_step, env_render, env_close,
//                vec_init, vec_reset, vec_step, vec_render, vec_close, vec_log
#include "../env_binding.h"

/* Custom initialization function
 * Called by env_binding.h to extract Python kwargs and initialize C env
 */
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // Extract parameters from Python kwargs
    env->width = (int)unpack(kwargs, "width");
    env->height = (int)unpack(kwargs, "height");
    env->num_agents = (int)unpack(kwargs, "num_agents");
    env->vision = (int)unpack(kwargs, "vision");
    env->bomb_timer = (int)unpack(kwargs, "bomb_timer");
    env->explosion_duration = (int)unpack(kwargs, "explosion_duration");
    env->default_blast_radius = (int)unpack(kwargs, "blast_radius");
    env->block_density = (float)unpack(kwargs, "block_density");
    env->max_ticks = (int)unpack(kwargs, "max_ticks");

    // Calculate derived array sizes for fixed allocations
    // These determine memory requirements and prevent overflow

    // max_bombs: Maximum simultaneous bombs across all agents
    // Formula: num_agents × 3 (each agent can have up to 3 bombs with power-ups)
    // Example: 8 agents × 3 = 24 bomb slots
    env->max_bombs = env->num_agents * 3;

    // max_fires: Maximum simultaneous fire cells from all explosions
    // Formula: max_bombs × 12 (each bomb creates ~12 fire cells: 4 directions × 3 radius)
    // Example: 24 bombs × 12 = 288 fire slots
    // TODO: This will need revision for battle royale feature (shrinking map creates permanent fires)
    env->max_fires = env->max_bombs * 12;

    // max_powerups: Maximum power-ups on ground at once (Phase 3 feature)
    // Formula: num_agents × 2 (spawn rate limits to ~2 power-ups per agent)
    // Example: 8 agents × 2 = 16 power-up slots
    env->max_powerups = env->num_agents * 2;

    // Initialize per-environment tick counter (C-side game state)
    // Python maintains a separate global tick for logging across all envs
    env->tick = 0;

    // Call C init function to allocate arrays
    init(env);

    // Check for Python errors during unpack
    if (PyErr_Occurred()) {
        return -1;
    }

    return 0;
}

/* Custom logging function
 * Called by env_binding.h's vec_log to populate Python dict
 */
static int my_log(PyObject* dict, Log* log) {
    // Export all log fields to Python dict
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
