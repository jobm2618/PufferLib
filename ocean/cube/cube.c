#include "cube.h"
#include <stdio.h>
#include <time.h>

static int check_failed = 0;

static void check(int cond, const char* msg) {
    if (!cond) {
        printf("FAIL: %s\n", msg);
        check_failed = 1;
    }
}

int main() {
    Cube env = {0};
    env.max_steps = 200;
    env.rng = 42;

    env.observations = (float*)calloc(CUBE_STICKERS, sizeof(float));
    env.actions      = (float*)calloc(1, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));

    // init no longer calls c_reset (pointers weren't set yet in vecenv mode)
    init(&env);
    c_reset(&env);

    uint8_t tmp[CUBE_STICKERS];

    // -------------------------------------------------------------------------
    // 1. Move + inverse = identity (all 18 moves)
    // -------------------------------------------------------------------------
    for (int m = 0; m < CUBE_MOVES; m++) {
        uint8_t state[CUBE_STICKERS];
        memcpy(state, SOLVED_STATE, CUBE_STICKERS);
        apply_move(state, m, tmp);
        apply_move(state, MOVE_INV[m], tmp);
        char buf[64];
        snprintf(buf, sizeof(buf), "move %d + MOVE_INV[%d] != identity", m, m);
        check(memcmp(state, SOLVED_STATE, CUBE_STICKERS) == 0, buf);
    }
    printf("Check 1 (move + inverse = identity): %s\n", check_failed ? "FAIL" : "OK");
    int prev_fail = check_failed;

    // -------------------------------------------------------------------------
    // 2. Any single move applied 4× = identity (quarter turns: 0,1,3,4,6,7,9,10,12,13,15,16)
    //    Half turns applied 2× = identity (2,5,8,11,14,17)
    // -------------------------------------------------------------------------
    for (int m = 0; m < CUBE_MOVES; m++) {
        uint8_t state[CUBE_STICKERS];
        memcpy(state, SOLVED_STATE, CUBE_STICKERS);
        int reps = (m % 3 == 2) ? 2 : 4;  // half-turn: 2x, quarter-turn: 4x
        for (int r = 0; r < reps; r++) apply_move(state, m, tmp);
        char buf[64];
        snprintf(buf, sizeof(buf), "move %d repeated %dx != identity", m, reps);
        check(memcmp(state, SOLVED_STATE, CUBE_STICKERS) == 0, buf);
    }
    printf("Check 2 (move cycle back to identity): %s\n",
           (check_failed != prev_fail) ? "FAIL" : "OK");
    prev_fail = check_failed;

    // -------------------------------------------------------------------------
    // 3. Solved state is_solved() = 1, scrambled = 0
    // -------------------------------------------------------------------------
    {
        uint8_t state[CUBE_STICKERS];
        memcpy(state, SOLVED_STATE, CUBE_STICKERS);
        check(is_solved(state) == 1, "SOLVED_STATE not detected as solved");
        apply_move(state, 0, tmp);  // U move
        check(is_solved(state) == 0, "moved cube detected as solved");
    }
    printf("Check 3 (is_solved): %s\n",
           (check_failed != prev_fail) ? "FAIL" : "OK");
    prev_fail = check_failed;

    // -------------------------------------------------------------------------
    // 4. count_correct on solved = 54, on scrambled < 54
    // -------------------------------------------------------------------------
    {
        uint8_t state[CUBE_STICKERS];
        memcpy(state, SOLVED_STATE, CUBE_STICKERS);
        check(count_correct(state) == 54, "count_correct(solved) != 54");
        apply_move(state, 0, tmp);
        check(count_correct(state) < 54, "count_correct(scrambled) == 54");
    }
    printf("Check 4 (count_correct): %s\n",
           (check_failed != prev_fail) ? "FAIL" : "OK");
    prev_fail = check_failed;

    // -------------------------------------------------------------------------
    // 5. Scramble pool: no entry should be solved (spot-check 256 entries)
    // -------------------------------------------------------------------------
    {
        int solved_count = 0;
        for (int p = 0; p < 256; p++)
            solved_count += is_solved(env.pool[p]);
        check(solved_count == 0, "scramble pool contains solved states");
    }
    printf("Check 5 (scramble pool not solved): %s\n",
           (check_failed != prev_fail) ? "FAIL" : "OK");
    prev_fail = check_failed;

    // -------------------------------------------------------------------------
    // 6. c_step increments step_count and sets terminal on truncation
    // -------------------------------------------------------------------------
    {
        c_reset(&env);
        int steps = 0;
        while (env.terminals[0] == 0.0f && steps < env.max_steps + 5) {
            env.actions[0] = 0.0f;  // always U
            c_step(&env);
            steps++;
        }
        // After max_steps, terminal should have fired and env auto-reset
        // step_count resets to 0 after terminal
        check(env.step_count == 0, "step_count not reset after terminal");
        check(steps <= env.max_steps + 1, "truncation did not fire on time");
    }
    printf("Check 6 (truncation + auto-reset): %s\n",
           (check_failed != prev_fail) ? "FAIL" : "OK");

    if (check_failed) {
        printf("\nSome checks FAILED.\n");
        return 1;
    }
    printf("\nAll checks passed.\n\n");

    // -------------------------------------------------------------------------
    // SPS benchmark (single env, 3 seconds)
    // -------------------------------------------------------------------------
    c_reset(&env);
    long long N = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double elapsed = 0;
    while (elapsed < 3.0) {
        env.actions[0] = (float)(rand_r(&env.rng) % CUBE_MOVES);
        c_step(&env);
        N++;
        if (N % 100000 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        }
    }
    printf("Cube SPS (1 env): %lld\n", (long long)(N / elapsed));

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
