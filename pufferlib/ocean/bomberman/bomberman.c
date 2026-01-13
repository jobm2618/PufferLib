/* Bomberman: Standalone demo and test functions
 * Compile standalone:
 *   gcc -o bomberman_demo bomberman.c -I../../.. -Iraylib-include -lraylib -lm
 *
 * Run demo:
 *   ./bomberman_demo
 */

#include <time.h>
#include "bomberman.h"

/* Demo function for interactive visualization
 */
int demo() {
    Bomberman env;

    // Initialize with reasonable defaults
    env.width = 21;
    env.height = 21;
    env.num_agents = 4;
    env.vision = 5;
    env.bomb_timer = 5;
    env.explosion_duration = 3;
    env.default_blast_radius = 2;
    env.block_density = 0.4f;
    env.max_ticks = 10000;

    // Calculate derived values (from binding.c logic)
    env.max_bombs = env.num_agents * 3;
    env.max_fires = env.max_bombs * 12;
    env.max_powerups = env.num_agents * 2;
    env.tick = 0;

    // Allocate memory
    init(&env);

    // Allocate action/reward arrays (not shared with Python in standalone mode)
    env.actions = (int*)calloc(env.num_agents, sizeof(int));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));
    env.observations = (float*)calloc(env.num_agents * TOTAL_OBS_SIZE, sizeof(float));

    // Reset environment
    c_reset(&env);

    // Render loop
    while (!WindowShouldClose()) {
        // Random actions for demo
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[i] = rand() % 9;
        }

        c_step(&env);
        c_render(&env);
    }

    // Cleanup
    c_close(&env);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    free(env.observations);

    return 0;
}

/* Performance benchmark
 * Note: bomberman.py already has test_performance(), this is for standalone C testing
 */
void test_performance(float test_time) {
    Bomberman env;

    // Large-scale benchmark config
    env.width = 23;
    env.height = 23;
    env.num_agents = 8;
    env.vision = 5;
    env.bomb_timer = 5;
    env.explosion_duration = 3;
    env.default_blast_radius = 2;
    env.block_density = 0.4f;
    env.max_ticks = 10000;

    // Calculate derived values
    env.max_bombs = env.num_agents * 3;
    env.max_fires = env.max_bombs * 12;
    env.max_powerups = env.num_agents * 2;
    env.tick = 0;

    // Allocate memory
    init(&env);
    env.actions = (int*)calloc(env.num_agents, sizeof(int));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));
    env.observations = (float*)calloc(env.num_agents * TOTAL_OBS_SIZE, sizeof(float));

    c_reset(&env);

    int start = time(NULL);
    int steps = 0;

    while (time(NULL) - start < test_time) {
        // Random actions
        for (int j = 0; j < env.num_agents; j++) {
            env.actions[j] = rand() % 9;
        }

        c_step(&env);
        steps++;
    }

    int end = time(NULL);
    float sps = (float)env.num_agents * steps / (end - start);

    printf("=== Bomberman Performance ===\n");
    printf("Config: %d agents on %dx%d grid\n", env.num_agents, env.width, env.height);
    printf("Steps: %d\n", steps);
    printf("Time: %d seconds\n", end - start);
    printf("SPS: %.0f steps/second\n", sps);

    // Cleanup
    c_close(&env);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    free(env.observations);
}

int main() {
    demo();
    // test_performance(10);  // Uncomment to run benchmark
    return 0;
}
