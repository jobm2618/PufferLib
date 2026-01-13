/* Battle royale like Bomberman: Multi-agent reinforcement learning environment
 * Supports 2-64 agents per environment with automatic grid scaling
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "raylib.h"

// Grid cell types (stored in grid array as char)
#define EMPTY 0
#define HARD_WALL 1
#define SOFT_WALL 2
// Values 3+ are used for agent IDs (AGENT_ID_OFFSET + agent_index)
#define AGENT_ID_OFFSET 3

// Observation channels (spatial)
#define NUM_SPATIAL_CHANNELS 8
#define CH_WALLS 0
#define CH_BLOCKS 1
#define CH_OTHERS 2            // Renumbered from 3
#define CH_BOMBS 3             // Renumbered from 4
#define CH_FIRES 4             // Renumbered from 5
#define CH_POWERUP_BOMB 5      // Bomb count power-up (Phase 3) - Renumbered from 6
#define CH_POWERUP_BLAST 6     // Blast radius power-up (Phase 3) - Renumbered from 7
#define CH_POWERUP_SPEED 7     // Speed power-up (Phase 3) - Renumbered from 8

// Scalar features (appended after spatial)
// Scalar features (4 values appended after spatial)
// Note: Survival progress removed - only needed for reward calculation, not observation
#define NUM_SCALAR_FEATURES 4
#define SCALAR_LIVES 0          // Remaining lives (normalized to 0.33, 0.67, 1.0)
#define SCALAR_BOMB_COUNT 1     // Current/max bombs
#define SCALAR_BLAST_RADIUS 2   // (current-2)/(max-2)
#define SCALAR_SPEED 3          // (speed-1.0)/(max-1.0)

// Total observation size (flattened)
#define SPATIAL_SIZE (11 * 11 * NUM_SPATIAL_CHANNELS)  // 968
#define TOTAL_OBS_SIZE (SPATIAL_SIZE + NUM_SCALAR_FEATURES)  // 972

// Actions
#define ACTION_NOOP 0
#define ACTION_UP 1
#define ACTION_DOWN 2
#define ACTION_LEFT 3
#define ACTION_RIGHT 4
#define ACTION_UP_BOMB 5
#define ACTION_DOWN_BOMB 6
#define ACTION_LEFT_BOMB 7
#define ACTION_RIGHT_BOMB 8

/* Required logging struct - only use floats!
 * These metrics are averaged across all environments
 * and reported to Python every log_interval ticks.
 */
typedef struct {
    float perf;              // Performance metric: (kills - deaths) / episode_length
    float score;             // Raw score: kills - deaths
    float episode_return;    // Sum of rewards over episode
    float episode_length;    // Number of steps in episode
    float n;                 // Required as the last field (episode count)
} Log;

/* Agent state
 * Tracks position, alive status, bomb inventory, and stats
 */
typedef struct {
    int x, y;                // Position in grid
    bool alive;              // true=alive, false=dead (0 lives remaining)
    int remaining_lives;     // 3, 2, 1, or 0 (0 = dead)
    int bomb_count;          // Current bombs available to place
    int max_bombs;           // Max simultaneous bombs (1-5 with power-ups)
    int blast_radius;        // Current blast radius (2-5 with power-ups)
    float speed_multiplier;  // Movement speed (1.0-2.0 with power-ups, Phase 3)
    int rank;                // Placement when eliminated (1=winner, N=first out)
} Agent;

/* Bomb state
 * Tracks bombs placed by agents
 * Uses fixed-size array with reuse (timer<0 = inactive slot)
 */
typedef struct {
    int x, y;                // Position in grid
    int timer;               // Ticks until explosion (-1 = inactive slot)
    int owner;               // Agent index who placed it
    int blast_radius;        // Explosion radius in cells
} Bomb;

/* Fire state
 * Tracks fire cells from explosions
 * Uses fixed-size array with reuse (remaining_ticks<0 = inactive slot)
 */
typedef struct {
    int x, y;                // Position in grid
    int remaining_ticks;     // How long fire persists (-1 = inactive slot)
    int owner;               // Agent index of bomb owner (for kill credit)
} Fire;

/* PowerUp state (Phase 3 feature, observation channels added now)
 * Tracks power-ups on the ground
 * Uses fixed-size array with reuse (active=0 = inactive slot)
 */
typedef struct {
    int x, y;                // Position in grid
    int type;                // 0=bomb_count, 1=blast_radius, 2=speed
    int active;              // 1=on ground, 0=collected/inactive
} PowerUp;

/* Raylib rendering client
 * Initialized on first render() call
 */
typedef struct {
    Texture2D agent_tex;     // Agent sprite
    Texture2D bomb_tex;      // Bomb sprite
    Texture2D fire_tex;      // Fire sprite
} Client;

/* Main environment struct
 * Contains all game state and shared buffers with Python
 *
 * Memory layout optimized for cache efficiency:
 * - Hot data (accessed every step) grouped together
 * - Configuration data (read-only) at end
 */
typedef struct {
    // === LOGGING (updated every step) ===
    Log log;                     // Aggregated metrics

    // === HOT DATA (cleared/written every step) ===
    float* rewards;              // Per-agent rewards (shared with Python)
    unsigned char* terminals;    // Per-agent episode end flags (shared with Python)
    int* actions;                // Per-agent actions (shared with Python, read-only)

    // === GAME STATE (read/write every step) ===
    Agent* agents;               // Per-agent state
    char* grid;                  // Flat grid: grid[y*width + x]
    Bomb* bombs;                 // Fixed-size bomb array
    Fire* fires;                 // Fixed-size fire array
    PowerUp* powerups;           // Fixed-size power-up array (Phase 3, allocated now)
    Log* agent_logs;             // Per-agent episode logs

    // === WARM DATA (written per-agent during obs computation) ===
    float* observations;         // Per-agent observations (shared with Python)

    // === COLD DATA (configuration, read-only after init) ===
    int width, height;           // Grid dimensions
    int num_agents;              // Number of agents
    int agents_alive;            // Count of agents with lives > 0
    int agents_eliminated;       // Count of eliminated agents (for ranking)
    int max_bombs;               // Size of bombs array
    int max_fires;               // Size of fires array
    int max_powerups;            // Size of power-ups array (Phase 3)
    int vision;                  // Vision radius (fixed: 5)
    int window;                  // Vision window size (2*vision+1 = 11)
    int obs_size;                // Observation size per agent (973 floats)
    int bomb_timer;              // Ticks until bomb explodes (fixed: 5)
    int explosion_duration;      // Ticks fire persists (fixed: 3)
    int default_blast_radius;    // Default explosion radius (fixed: 2)
    float block_density;         // Fraction of map filled with soft blocks (fixed: 0.4)
    int tick;                    // Current game tick
    int max_ticks;               // Maximum ticks per episode (for battle royale)

    // === RENDERING (only used if render() called) ===
    Client* client;              // Raylib client (NULL until first render)
} Bomberman;

// === FUNCTION DECLARATIONS ===

/* Initialize environment (called once after struct creation)
 * Allocates memory for agents, bombs, fires, grid
 */
void init(Bomberman* env);

/* Reset environment to initial state (called at episode start)
 * Respawns agents, clears bombs/fires, regenerates map
 */
void c_reset(Bomberman* env);

/* Execute one step of the game
 * Reads actions from env->actions
 * Writes results to env->observations, rewards, terminals
 */
void c_step(Bomberman* env);

/* Render environment to screen (Raylib)
 * Only renders if client is initialized
 */
void c_render(Bomberman* env);

/* Clean up allocated memory
 * Does NOT free env->observations, actions, rewards, terminals
 * (those are managed by Python)
 */
void c_close(Bomberman* env);

/* Compute observations for all agents
 * Extracts local (window × window) view for each agent
 * Normalizes all values to [0, 1] range
 */
void compute_observations(Bomberman* env);

// === HELPER FUNCTIONS ===

/* Place a bomb at agent's position
 * Returns true if successful, false if no bombs available
 */
bool place_bomb(Bomberman* env, int agent_idx);

/* Trigger explosion at bomb position
 * Creates fire cells in 4 directions, destroys soft walls, kills agents
 */
void explode_bomb(Bomberman* env, int bomb_idx);

/* Check if agent position has active fire
 * Returns owner of fire if present, -1 if no fire
 */
int check_fire_collision(Bomberman* env, int x, int y);

/* Damage agent (3-lives system)
 * Fire contact: -1 life
 * At 0 lives: mark dead, give placement reward, set terminal
 */
void damage_agent(Bomberman* env, int agent_idx, int killer_idx);

/* Spawn agent at random empty position
 * Used at reset and respawn (if implemented)
 */
void spawn_agent(Bomberman* env, int agent_idx);

/* Generate random map with soft blocks
 * Border is hard walls, interior has random soft blocks
 */
void generate_map(Bomberman* env);

// === IMPLEMENTATION ===
// Implementation in header (PufferLib pattern for single-file envs)

/* === INITIALIZATION === */

void init(Bomberman* env) {
    // Allocate game state arrays
    env->agents = (Agent*)calloc(env->num_agents, sizeof(Agent));
    env->agent_logs = (Log*)calloc(env->num_agents, sizeof(Log));
    env->grid = (char*)calloc(env->width * env->height, sizeof(char));
    env->bombs = (Bomb*)calloc(env->max_bombs, sizeof(Bomb));
    env->fires = (Fire*)calloc(env->max_fires, sizeof(Fire));
    env->powerups = (PowerUp*)calloc(env->max_powerups, sizeof(PowerUp));

    // Initialize computed values
    env->window = 2 * env->vision + 1;  // 11
    env->obs_size = TOTAL_OBS_SIZE;  // 973 (11×11×8 + 5 scalars)

    // Initialize counters
    env->agents_alive = env->num_agents;
    env->agents_eliminated = 0;

    // Mark all bombs/fires/powerups as inactive
    for (int i = 0; i < env->max_bombs; i++) {
        env->bombs[i].timer = -1;
    }
    for (int i = 0; i < env->max_fires; i++) {
        env->fires[i].remaining_ticks = -1;
    }
    for (int i = 0; i < env->max_powerups; i++) {
        env->powerups[i].active = 0;
    }

    // Initialize rendering client as NULL (created on first render)
    env->client = NULL;
}

/* === MAP GENERATION === */

void generate_map(Bomberman* env) {
    // Clear grid
    memset(env->grid, EMPTY, env->width * env->height);

    // Create border walls (hard)
    for (int x = 0; x < env->width; x++) {
        env->grid[0 * env->width + x] = HARD_WALL;  // Top
        env->grid[(env->height - 1) * env->width + x] = HARD_WALL;  // Bottom
    }
    for (int y = 0; y < env->height; y++) {
        env->grid[y * env->width + 0] = HARD_WALL;  // Left
        env->grid[y * env->width + (env->width - 1)] = HARD_WALL;  // Right
    }

    // Place soft blocks with symmetric mirroring (fairness for all spawn corners)
    // Algorithm: randomly place blocks, mirror to 3 other positions
    int num_interior_cells = (env->width - 2) * (env->height - 2);
    int num_blocks = (int)(num_interior_cells * env->block_density);

    int blocks_placed = 0;
    int max_attempts = num_blocks * 10;  // Prevent infinite loop
    int attempts = 0;

    while (blocks_placed < num_blocks && attempts < max_attempts) {
        attempts++;

        // Randomly select position (x, y) in interior
        int x = 1 + rand() % (env->width - 2);
        int y = 1 + rand() % (env->height - 2);

        // Calculate mirrored positions
        int x_h = env->width - 1 - x;   // Horizontal flip
        int y_v = env->height - 1 - y;  // Vertical flip
        int x_d = x_h;                  // Diagonal flip (both)
        int y_d = y_v;

        // Check if all 4 positions are valid (not already occupied, not spawn corners)
        bool valid = true;
        int positions[4][2] = {{x, y}, {x_h, y}, {x, y_v}, {x_d, y_d}};

        for (int i = 0; i < 4; i++) {
            int px = positions[i][0];
            int py = positions[i][1];
            int idx = py * env->width + px;

            // Check if already occupied
            if (env->grid[idx] != EMPTY) {
                valid = false;
                break;
            }

            // Check spawn corners (keep 2×2 area clear at corners)
            bool is_spawn_corner = false;
            if (px <= 2 && py <= 2) is_spawn_corner = true;  // Top-left
            if (px >= env->width - 3 && py <= 2) is_spawn_corner = true;  // Top-right
            if (px <= 2 && py >= env->height - 3) is_spawn_corner = true;  // Bottom-left
            if (px >= env->width - 3 && py >= env->height - 3) is_spawn_corner = true;  // Bottom-right

            if (is_spawn_corner) {
                valid = false;
                break;
            }

            // Check if it's a hard wall (shouldn't happen with interior range, but safety check)
            if (env->grid[idx] == HARD_WALL) {
                valid = false;
                break;
            }
        }

        // If all positions valid, place blocks at all 4 mirrored locations
        if (valid) {
            for (int i = 0; i < 4; i++) {
                int px = positions[i][0];
                int py = positions[i][1];
                int idx = py * env->width + px;
                env->grid[idx] = SOFT_WALL;
            }
            blocks_placed += 4;  // Placed 4 blocks (or fewer if some overlap at center)
        }
    }

    // Note: blocks_placed may exceed num_blocks slightly due to 4-block placement
    // This is acceptable for fairness - ensures all corners have identical patterns
}

/* === AGENT MANAGEMENT === */

void spawn_agent(Bomberman* env, int agent_idx) {
    Agent* agent = &env->agents[agent_idx];

    // Spawn in corners (distribute evenly)
    // For 4 agents: corners. For 8 agents: corners + midpoints. Etc.
    int spawn_idx = agent_idx % 4;
    int x, y;

    switch (spawn_idx) {
        case 0:  // Top-left
            x = 1;
            y = 1;
            break;
        case 1:  // Top-right
            x = env->width - 2;
            y = 1;
            break;
        case 2:  // Bottom-left
            x = 1;
            y = env->height - 2;
            break;
        case 3:  // Bottom-right
            x = env->width - 2;
            y = env->height - 2;
            break;
    }

    // Offset slightly for multiple agents at same corner
    int corner_offset = agent_idx / 4;
    if (corner_offset > 0) {
        x += (corner_offset % 2) * 2 - 1;  // Wiggle x
        y += (corner_offset / 2) * 2 - 1;  // Wiggle y

        // Clamp to valid range
        x = x < 1 ? 1 : (x >= env->width - 1 ? env->width - 2 : x);
        y = y < 1 ? 1 : (y >= env->height - 1 ? env->height - 2 : y);
    }

    agent->x = x;
    agent->y = y;
    agent->alive = true;
    agent->remaining_lives = 3;  // Start with 3 lives
    agent->bomb_count = 1;  // Start with 1 bomb
    agent->max_bombs = 1;
    agent->blast_radius = env->default_blast_radius;  // 2
    agent->speed_multiplier = 1.0f;  // Normal speed
    agent->rank = 0;  // Not yet eliminated

    // Clear agent's episode log
    env->agent_logs[agent_idx] = (Log){0};
}

/* === BOMB MANAGEMENT === */

bool place_bomb(Bomberman* env, int agent_idx) {
    Agent* agent = &env->agents[agent_idx];

    // Check if agent has bombs available
    if (agent->bomb_count <= 0) return false;

    // Find inactive bomb slot
    int bomb_idx = -1;
    for (int i = 0; i < env->max_bombs; i++) {
        if (env->bombs[i].timer < 0) {
            bomb_idx = i;
            break;
        }
    }

    // No free slots (should be rare with max_bombs = num_agents * 3)
    if (bomb_idx < 0) return false;

    // Place bomb at agent position
    Bomb* bomb = &env->bombs[bomb_idx];
    bomb->x = agent->x;
    bomb->y = agent->y;
    bomb->timer = env->bomb_timer;
    bomb->owner = agent_idx;
    bomb->blast_radius = agent->blast_radius;

    // Decrement agent's available bombs
    agent->bomb_count--;

    return true;
}

void explode_bomb(Bomberman* env, int bomb_idx) {
    Bomb* bomb = &env->bombs[bomb_idx];
    int center_x = bomb->x;
    int center_y = bomb->y;
    int radius = bomb->blast_radius;
    int owner = bomb->owner;

    // Deactivate bomb
    bomb->timer = -1;

    // Return bomb to owner's inventory
    if (owner >= 0 && owner < env->num_agents) {
        env->agents[owner].bomb_count++;
    }

    // Directions: up, down, left, right
    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};

    // Propagate fire in 4 directions
    for (int dir = 0; dir < 4; dir++) {
        for (int dist = 0; dist <= radius; dist++) {
            int x = center_x + dx[dir] * dist;
            int y = center_y + dy[dir] * dist;

            // Check bounds
            if (x < 0 || x >= env->width || y < 0 || y >= env->height) break;

            int grid_idx = y * env->width + x;
            char cell = env->grid[grid_idx];

            // Stop at hard walls
            if (cell == HARD_WALL) break;

            // Destroy soft walls and stop propagation
            if (cell == SOFT_WALL) {
                env->grid[grid_idx] = EMPTY;
                // No reward for block destruction - only placement + survival rewards at elimination
                break;  // Don't propagate through blocks
            }

            // Create fire at this cell
            // Find inactive fire slot
            for (int f = 0; f < env->max_fires; f++) {
                if (env->fires[f].remaining_ticks < 0) {
                    env->fires[f].x = x;
                    env->fires[f].y = y;
                    env->fires[f].remaining_ticks = env->explosion_duration;
                    env->fires[f].owner = owner;
                    break;
                }
            }

            // Chain reaction: trigger other bombs
            for (int b = 0; b < env->max_bombs; b++) {
                if (env->bombs[b].timer >= 0 &&
                    env->bombs[b].x == x && env->bombs[b].y == y) {
                    explode_bomb(env, b);  // Recursive!
                }
            }
        }
    }
}

/* === COLLISION DETECTION === */

int check_fire_collision(Bomberman* env, int x, int y) {
    for (int i = 0; i < env->max_fires; i++) {
        if (env->fires[i].remaining_ticks > 0 &&
            env->fires[i].x == x && env->fires[i].y == y) {
            return env->fires[i].owner;
        }
    }
    return -1;
}

void damage_agent(Bomberman* env, int agent_idx, int killer_idx) {
    Agent* agent = &env->agents[agent_idx];

    // Skip if already dead
    if (!agent->alive) return;

    // Lose 1 life
    agent->remaining_lives--;

    // Check if agent is eliminated (0 lives remaining)
    if (agent->remaining_lives <= 0) {
        // Mark as dead
        agent->alive = false;
        env->terminals[agent_idx] = 1;
        env->agents_alive--;

        // Calculate placement (reverse rank)
        // First eliminated = rank N (worst), last surviving = rank 1 (winner)
        agent->rank = env->num_agents - env->agents_eliminated;
        env->agents_eliminated++;

        // Placement reward (linear): (N - rank) / N
        // Example: Rank 1 of 8 → (8-1)/8 = 0.875 (winner)
        // Example: Rank 8 of 8 → (8-8)/8 = 0.0 (first out)
        float placement_reward = (float)(env->num_agents - agent->rank) / (float)env->num_agents;

        // Survival reward: (agent's steps survived) / (total game ticks so far)
        // Uses agent's episode_length for steps, env->tick for game ticks
        // Example: Agent survived 100 steps in a 150-tick game → 100/150 = 0.67
        Log* agent_log = &env->agent_logs[agent_idx];
        float survival_reward = (float)agent_log->episode_length / (float)env->tick;

        // Total reward at elimination
        float total_reward = placement_reward + survival_reward;
        env->rewards[agent_idx] += total_reward;
        env->agent_logs[agent_idx].episode_return += total_reward;

        // Update episode log
        Log* log = &env->agent_logs[agent_idx];
        log->score = total_reward;  // Use total reward as score
        log->perf = log->episode_length > 0 ? log->score / log->episode_length : 0.0f;

        // Add to aggregate log
        env->log.perf += log->perf;
        env->log.score += log->score;
        env->log.episode_return += log->episode_return;
        env->log.episode_length += log->episode_length;
        env->log.n += 1;
    }
    // If lives > 0, agent survives this hit (optional: add brief invulnerability here)
}

/* === OBSERVATION COMPUTATION === */

void compute_observations(Bomberman* env) {
    int obs_idx = 0;

    for (int a = 0; a < env->num_agents; a++) {
        Agent* agent = &env->agents[a];

        // Skip dead agents (HUGE performance savings)
        if (!agent->alive) {
            obs_idx += env->obs_size;
            continue;
        }

        int center_x = agent->x;
        int center_y = agent->y;

        // === SPATIAL OBSERVATIONS (11×11×8 = 968 floats) ===
        // Extract local vision window
        for (int dy = -env->vision; dy <= env->vision; dy++) {
            for (int dx = -env->vision; dx <= env->vision; dx++) {
                int x = center_x + dx;
                int y = center_y + dy;

                // Out of bounds = walls in channel 0, zeros in other channels
                if (x < 0 || x >= env->width || y < 0 || y >= env->height) {
                    env->observations[obs_idx++] = 1.0f;  // CH_WALLS
                    env->observations[obs_idx++] = 0.0f;  // CH_BLOCKS
                    env->observations[obs_idx++] = 0.0f;  // CH_OTHERS (now channel 2)
                    env->observations[obs_idx++] = 0.0f;  // CH_BOMBS (now channel 3)
                    env->observations[obs_idx++] = 0.0f;  // CH_FIRES (now channel 4)
                    env->observations[obs_idx++] = 0.0f;  // CH_POWERUP_BOMB (now channel 5)
                    env->observations[obs_idx++] = 0.0f;  // CH_POWERUP_BLAST (now channel 6)
                    env->observations[obs_idx++] = 0.0f;  // CH_POWERUP_SPEED (now channel 7)
                    continue;
                }

                int grid_idx = y * env->width + x;
                char cell = env->grid[grid_idx];

                // Channel 0: Hard walls
                env->observations[obs_idx++] = (cell == HARD_WALL) ? 1.0f : 0.0f;

                // Channel 1: Soft blocks
                env->observations[obs_idx++] = (cell == SOFT_WALL) ? 1.0f : 0.0f;

                // Channel 2: Other agents (normalized by num_agents)
                float other_agent_val = 0.0f;
                for (int other_a = 0; other_a < env->num_agents; other_a++) {
                    if (other_a != a && env->agents[other_a].alive &&
                        env->agents[other_a].x == x && env->agents[other_a].y == y) {
                        other_agent_val = 1.0f / (float)env->num_agents;
                        break;
                    }
                }
                env->observations[obs_idx++] = other_agent_val;

                // Channel 3: Bombs (timer normalized to [0,1])
                float bomb_val = 0.0f;
                for (int b = 0; b < env->max_bombs; b++) {
                    if (env->bombs[b].timer > 0 &&
                        env->bombs[b].x == x && env->bombs[b].y == y) {
                        bomb_val = (float)env->bombs[b].timer / (float)env->bomb_timer;
                        break;
                    }
                }
                env->observations[obs_idx++] = bomb_val;

                // Channel 4: Fires
                float fire_val = 0.0f;
                for (int f = 0; f < env->max_fires; f++) {
                    if (env->fires[f].remaining_ticks > 0 &&
                        env->fires[f].x == x && env->fires[f].y == y) {
                        fire_val = 1.0f;
                        break;
                    }
                }
                env->observations[obs_idx++] = fire_val;

                // Channels 5-7: Power-ups (Phase 3 - currently always 0)
                float powerup_bomb = 0.0f;
                float powerup_blast = 0.0f;
                float powerup_speed = 0.0f;
                // TODO Phase 3: Check env->powerups array for power-ups at (x, y)
                env->observations[obs_idx++] = powerup_bomb;
                env->observations[obs_idx++] = powerup_blast;
                env->observations[obs_idx++] = powerup_speed;
            }
        }

        // === SCALAR FEATURES (4 floats appended after spatial) ===
        // Note: Survival progress removed - only needed for reward calculation

        // 0: Remaining lives (normalized to 0.33, 0.67, 1.0)
        env->observations[obs_idx++] = (float)agent->remaining_lives / 3.0f;

        // 1: Bomb count (current/max)
        env->observations[obs_idx++] = (float)agent->bomb_count / (float)agent->max_bombs;

        // 2: Blast radius ((current-2)/(max-2)) - baseline is 2, max is 5 (with power-ups)
        // Range: 2-5 → normalized to [0, 1]
        float blast_normalized = (float)(agent->blast_radius - 2) / 3.0f;
        env->observations[obs_idx++] = blast_normalized;

        // 3: Speed multiplier ((speed-1.0)/(max-1.0)) - baseline is 1.0, max is 2.0
        // Range: 1.0-2.0 → normalized to [0, 1]
        float speed_normalized = (agent->speed_multiplier - 1.0f) / 1.0f;
        env->observations[obs_idx++] = speed_normalized;
    }
}

/* === RESET === */

void c_reset(Bomberman* env) {
    // Reset per-environment tick (C-side game state)
    // This is independent from Python's global tick counter
    env->tick = 0;
    env->log = (Log){0};

    // Reset counters
    env->agents_alive = env->num_agents;
    env->agents_eliminated = 0;

    // Clear agent logs
    for (int i = 0; i < env->num_agents; i++) {
        env->agent_logs[i] = (Log){0};
    }

    // Generate map (with symmetric block placement)
    generate_map(env);

    // Spawn all agents
    for (int i = 0; i < env->num_agents; i++) {
        spawn_agent(env, i);
    }

    // Clear bombs, fires, and power-ups
    for (int i = 0; i < env->max_bombs; i++) {
        env->bombs[i].timer = -1;
    }
    for (int i = 0; i < env->max_fires; i++) {
        env->fires[i].remaining_ticks = -1;
    }
    for (int i = 0; i < env->max_powerups; i++) {
        env->powerups[i].active = 0;
    }

    // Clear rewards and terminals
    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0;
    }

    // Compute initial observations
    compute_observations(env);
}

/* === STEP === */

void c_step(Bomberman* env) {
    env->tick++;

    // 1. Clear per-step state (no per-step rewards)
    for (int a = 0; a < env->num_agents; a++) {
        env->rewards[a] = 0.0f;
        env->terminals[a] = 0;

        // Increment episode length for alive agents
        if (env->agents[a].alive) {
            env->agent_logs[a].episode_length += 1;
        }
    }

    // 2. Process agent actions
    for (int a = 0; a < env->num_agents; a++) {
        if (!env->agents[a].alive) continue;

        Agent* agent = &env->agents[a];
        int action = env->actions[a];

        // Extract movement and bomb components
        int move_dir = -1;
        bool place_bomb_action = false;

        switch (action) {
            case ACTION_NOOP:
                break;
            case ACTION_UP:
                move_dir = 0;
                break;
            case ACTION_DOWN:
                move_dir = 1;
                break;
            case ACTION_LEFT:
                move_dir = 2;
                break;
            case ACTION_RIGHT:
                move_dir = 3;
                break;
            case ACTION_UP_BOMB:
                move_dir = 0;
                place_bomb_action = true;
                break;
            case ACTION_DOWN_BOMB:
                move_dir = 1;
                place_bomb_action = true;
                break;
            case ACTION_LEFT_BOMB:
                move_dir = 2;
                place_bomb_action = true;
                break;
            case ACTION_RIGHT_BOMB:
                move_dir = 3;
                place_bomb_action = true;
                break;
        }

        // Movement
        if (move_dir >= 0) {
            int dx[] = {0, 0, -1, 1};
            int dy[] = {-1, 1, 0, 0};
            int new_x = agent->x + dx[move_dir];
            int new_y = agent->y + dy[move_dir];

            // Check bounds and collisions
            if (new_x >= 0 && new_x < env->width &&
                new_y >= 0 && new_y < env->height) {
                int grid_idx = new_y * env->width + new_x;
                char cell = env->grid[grid_idx];

                // Can move if empty or has fire (will die from fire later)
                if (cell == EMPTY) {
                    agent->x = new_x;
                    agent->y = new_y;
                }
            }
        }

        // Bomb placement
        if (place_bomb_action) {
            place_bomb(env, a);
        }
    }

    // 3. Update bombs
    for (int b = 0; b < env->max_bombs; b++) {
        if (env->bombs[b].timer < 0) continue;

        env->bombs[b].timer--;
        if (env->bombs[b].timer == 0) {
            explode_bomb(env, b);
        }
    }

    // 4. Update fires
    for (int f = 0; f < env->max_fires; f++) {
        if (env->fires[f].remaining_ticks < 0) continue;

        env->fires[f].remaining_ticks--;
    }

    // 5. Check collisions (agent in fire) - 3-lives system
    for (int a = 0; a < env->num_agents; a++) {
        if (!env->agents[a].alive) continue;

        Agent* agent = &env->agents[a];
        int killer = check_fire_collision(env, agent->x, agent->y);
        if (killer >= 0) {
            damage_agent(env, a, killer);
        }
    }

    // 6. Compute observations
    compute_observations(env);
}

/* === RENDERING === */

// Colors (PufferLib standard palette)
const Color PUFF_BG = (Color){6, 24, 24, 255};
const Color PUFF_WALL = (Color){128, 128, 128, 255};
const Color PUFF_BLOCK = (Color){139, 69, 19, 255};  // Brown
const Color PUFF_FIRE = (Color){255, 69, 0, 255};  // Red-orange
const Color PUFF_BOMB = (Color){32, 32, 32, 255};  // Dark gray
const Color PUFF_AGENTS[] = {
    (Color){255, 0, 0, 255},     // Red
    (Color){0, 0, 255, 255},     // Blue
    (Color){0, 255, 0, 255},     // Green
    (Color){255, 255, 0, 255},   // Yellow
    (Color){255, 0, 255, 255},   // Magenta
    (Color){0, 255, 255, 255},   // Cyan
    (Color){255, 128, 0, 255},   // Orange
    (Color){128, 0, 255, 255},   // Purple
};

void c_render(Bomberman* env) {
    // Initialize window on first render
    if (env->client == NULL) {
        int window_width = env->width * 32;
        int window_height = env->height * 32;
        InitWindow(window_width, window_height, "PufferLib Bomberman");
        SetTargetFPS(10);
        env->client = (Client*)calloc(1, sizeof(Client));
    }

    // ESC to exit
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BG);

    int cell_size = 32;

    // Draw grid
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int grid_idx = y * env->width + x;
            char cell = env->grid[grid_idx];

            Color color = PUFF_BG;
            if (cell == HARD_WALL) color = PUFF_WALL;
            else if (cell == SOFT_WALL) color = PUFF_BLOCK;

            if (cell != EMPTY) {
                DrawRectangle(x * cell_size, y * cell_size, cell_size, cell_size, color);
            }
        }
    }

    // Draw fires
    for (int f = 0; f < env->max_fires; f++) {
        if (env->fires[f].remaining_ticks > 0) {
            DrawRectangle(
                env->fires[f].x * cell_size,
                env->fires[f].y * cell_size,
                cell_size, cell_size, PUFF_FIRE
            );
        }
    }

    // Draw bombs
    for (int b = 0; b < env->max_bombs; b++) {
        if (env->bombs[b].timer > 0) {
            DrawCircle(
                env->bombs[b].x * cell_size + cell_size / 2,
                env->bombs[b].y * cell_size + cell_size / 2,
                cell_size / 3, PUFF_BOMB
            );
        }
    }

    // Draw agents
    for (int a = 0; a < env->num_agents; a++) {
        if (env->agents[a].alive) {
            Color agent_color = PUFF_AGENTS[a % 8];
            DrawCircle(
                env->agents[a].x * cell_size + cell_size / 2,
                env->agents[a].y * cell_size + cell_size / 2,
                cell_size / 2.5f, agent_color
            );
        }
    }

    EndDrawing();
}

/* === CLEANUP === */

void c_close(Bomberman* env) {
    free(env->agents);
    free(env->agent_logs);
    free(env->grid);
    free(env->bombs);
    free(env->fires);

    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
    }
}
