#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

#define CUBE_STICKERS      54
#define CUBE_FACES          6
#define CUBE_FACE_SIZE      9
#define CUBE_MOVES         18
#define SCRAMBLE_MOVES     26
#define SCRAMBLE_POOL_SIZE 8192
#define EXTRA_MOVES         3

// ---------------------------------------------------------------------------
// Compile-time tables
// ---------------------------------------------------------------------------
// Face layout (looking at face from outside, row-major):
//   0 1 2
//   3 4 5
//   6 7 8
// Face offsets: U=0, D=9, F=18, B=27, L=36, R=45
//
// MOVE_PERM[m][i] = source sticker index:  new_cube[i] = old_cube[MOVE_PERM[m][i]]
// Move order: U,U',U2, D,D',D2, F,F',F2, B,B',B2, L,L',L2, R,R',R2

static const uint8_t MOVE_PERM[18][54] = {
    // 0: U
    {
     6, 3, 0, 7, 4, 1, 8, 5, 2,
     9,10,11,12,13,14,15,16,17,
    36,37,38,21,22,23,24,25,26,
    27,28,29,30,31,32,47,46,45,
    35,34,33,39,40,41,42,43,44,
    18,19,20,48,49,50,51,52,53
    },
    // 1: U'
    {
     2, 5, 8, 1, 4, 7, 0, 3, 6,
     9,10,11,12,13,14,15,16,17,
    45,46,47,21,22,23,24,25,26,
    27,28,29,30,31,32,38,37,36,
    18,19,20,39,40,41,42,43,44,
    35,34,33,48,49,50,51,52,53
    },
    // 2: U2
    {
     8, 7, 6, 5, 4, 3, 2, 1, 0,
     9,10,11,12,13,14,15,16,17,
    35,34,33,21,22,23,24,25,26,
    27,28,29,30,31,32,20,19,18,
    45,46,47,39,40,41,42,43,44,
    36,37,38,48,49,50,51,52,53
    },
    // 3: D
    {
     0, 1, 2, 3, 4, 5, 6, 7, 8,
    15,12, 9,16,13,10,17,14,11,
    18,19,20,21,22,23,51,52,53,
    44,43,42,30,31,32,33,34,35,
    36,37,38,39,40,41,24,25,26,
    45,46,47,48,49,50,29,28,27
    },
    // 4: D'
    {
     0, 1, 2, 3, 4, 5, 6, 7, 8,
    11,14,17,10,13,16, 9,12,15,
    18,19,20,21,22,23,42,43,44,
    53,52,51,30,31,32,33,34,35,
    36,37,38,39,40,41,29,28,27,
    45,46,47,48,49,50,24,25,26
    },
    // 5: D2
    {
     0, 1, 2, 3, 4, 5, 6, 7, 8,
    17,16,15,14,13,12,11,10, 9,
    18,19,20,21,22,23,29,28,27,
    26,25,24,30,31,32,33,34,35,
    36,37,38,39,40,41,51,52,53,
    45,46,47,48,49,50,42,43,44
    },
    // 6: F
    {
     0, 1, 2, 3, 4, 5,38,41,44,
    51,48,45,12,13,14,15,16,17,
    24,21,18,25,22,19,26,23,20,
    27,28,29,30,31,32,33,34,35,
    36,37,11,39,40,10,42,43, 9,
     6,46,47, 7,49,50, 8,52,53
    },
    // 7: F'
    {
     0, 1, 2, 3, 4, 5,45,48,51,
    44,41,38,12,13,14,15,16,17,
    20,23,26,19,22,25,18,21,24,
    27,28,29,30,31,32,33,34,35,
    36,37, 6,39,40, 7,42,43, 8,
    11,46,47,10,49,50, 9,52,53
    },
    // 8: F2
    {
     0, 1, 2, 3, 4, 5,11,10, 9,
     8, 7, 6,12,13,14,15,16,17,
    26,25,24,23,22,21,20,19,18,
    27,28,29,30,31,32,33,34,35,
    36,37,45,39,40,48,42,43,51,
    38,46,47,41,49,50,44,52,53
    },
    // 9: B
    {
    53,50,47, 3, 4, 5, 6, 7, 8,
     9,10,11,12,13,14,36,39,42,
    18,19,20,21,22,23,24,25,26,
    33,30,27,34,31,28,35,32,29,
     2,37,38, 1,40,41, 0,43,44,
    45,46,15,48,49,16,51,52,17
    },
    // 10: B'
    {
    42,39,36, 3, 4, 5, 6, 7, 8,
     9,10,11,12,13,14,47,50,53,
    18,19,20,21,22,23,24,25,26,
    29,32,35,28,31,34,27,30,33,
    15,37,38,16,40,41,17,43,44,
    45,46, 2,48,49, 1,51,52, 0
    },
    // 11: B2
    {
    17,16,15, 3, 4, 5, 6, 7, 8,
     9,10,11,12,13,14, 2, 1, 0,
    18,19,20,21,22,23,24,25,26,
    35,34,33,32,31,30,29,28,27,
    47,37,38,50,40,41,53,43,44,
    45,46,36,48,49,39,51,52,42
    },
    // 12: L
    {
    35, 1, 2,32, 4, 5,29, 7, 8,
    18,10,11,21,13,14,24,16,17,
     0,19,20, 3,22,23, 6,25,26,
    27,28,15,30,31,12,33,34, 9,
    42,39,36,43,40,37,44,41,38,
    45,46,47,48,49,50,51,52,53
    },
    // 13: L'
    {
    18, 1, 2,21, 4, 5,24, 7, 8,
    35,10,11,32,13,14,29,16,17,
     9,19,20,12,22,23,15,25,26,
    27,28, 6,30,31, 3,33,34, 0,
    38,41,44,37,40,43,36,39,42,
    45,46,47,48,49,50,51,52,53
    },
    // 14: L2
    {
     9, 1, 2,12, 4, 5,15, 7, 8,
     0,10,11, 3,13,14, 6,16,17,
    35,19,20,32,22,23,29,25,26,
    27,28,24,30,31,21,33,34,18,
    44,43,42,41,40,39,38,37,36,
    45,46,47,48,49,50,51,52,53
    },
    // 15: R
    {
     0, 1,20, 3, 4,23, 6, 7,26,
     9,10,33,12,13,30,15,16,27,
    18,19,11,21,22,14,24,25,17,
     8,28,29, 5,31,32, 2,34,35,
    36,37,38,39,40,41,42,43,44,
    51,48,45,52,49,46,53,50,47
    },
    // 16: R'
    {
     0, 1,33, 3, 4,30, 6, 7,27,
     9,10,20,12,13,23,15,16,26,
    18,19, 2,21,22, 5,24,25, 8,
    17,28,29,14,31,32,11,34,35,
    36,37,38,39,40,41,42,43,44,
    47,50,53,46,49,52,45,48,51
    },
    // 17: R2
    {
     0, 1,11, 3, 4,14, 6, 7,17,
     9,10, 2,12,13, 5,15,16, 8,
    18,19,33,21,22,30,24,25,27,
    26,28,29,23,31,32,20,34,35,
    36,37,38,39,40,41,42,43,44,
    53,52,51,50,49,48,47,46,45
    }
};

// Inverse move: MOVE_INV[m] undoes move m
static const uint8_t MOVE_INV[18] = {
    1, 0, 2,   // U, U', U2
    4, 3, 5,   // D, D', D2
    7, 6, 8,   // F, F', F2
   10, 9,11,   // B, B', B2
   13,12,14,   // L, L', L2
   16,15,17,   // R, R', R2
};

// Float color lookup: avoids per-sticker division
static const float FLOAT_COLOR[6] = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};

// Solved state for fast memcmp check
static const uint8_t SOLVED_STATE[54] = {
    0,0,0,0,0,0,0,0,0,  // U
    1,1,1,1,1,1,1,1,1,  // D
    2,2,2,2,2,2,2,2,2,  // F
    3,3,3,3,3,3,3,3,3,  // B
    4,4,4,4,4,4,4,4,4,  // L
    5,5,5,5,5,5,5,5,5,  // R
};

// ---------------------------------------------------------------------------
// Log struct
// ---------------------------------------------------------------------------
typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float solve_rate;
    float correct_at_terminal;
    float n;
};

// ---------------------------------------------------------------------------
// Cube struct
// ---------------------------------------------------------------------------
typedef struct Cube Cube;
struct Cube {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    Log log;
    int num_agents;
    unsigned int rng;

    uint8_t cube[CUBE_STICKERS];
    uint8_t tmp[CUBE_STICKERS];

    uint8_t (*pool)[CUBE_STICKERS];

    int step_count;
    int max_steps;
};

// ---------------------------------------------------------------------------
// Core inline functions
// ---------------------------------------------------------------------------
static inline void apply_move(uint8_t* cube, int m, uint8_t* tmp) {
    const uint8_t* p = MOVE_PERM[m];
    for (int i = 0; i < CUBE_STICKERS; i++) tmp[i] = cube[p[i]];
    memcpy(cube, tmp, CUBE_STICKERS);
}

static inline int is_solved(const uint8_t* cube) {
    return memcmp(cube, SOLVED_STATE, CUBE_STICKERS) == 0;
}

static inline int count_correct(const uint8_t* cube) {
    int n = 0;
    for (int f = 0; f < CUBE_FACES; f++) {
        uint8_t c = cube[f*9+4];
        for (int i = 0; i < CUBE_FACE_SIZE; i++)
            n += (cube[f*9+i] == c);
    }
    return n;
}

static inline void update_obs(Cube* env) {
    for (int i = 0; i < CUBE_STICKERS; i++)
        env->observations[i] = FLOAT_COLOR[env->cube[i]];
}

static void build_pool(Cube* env) {
    for (int p = 0; p < SCRAMBLE_POOL_SIZE; p++) {
        memcpy(env->pool[p], SOLVED_STATE, CUBE_STICKERS);
        int last_face = -1;
        for (int m = 0; m < SCRAMBLE_MOVES; m++) {
            int move;
            do { move = rand_r(&env->rng) % CUBE_MOVES; } while (move/3 == last_face);
            apply_move(env->pool[p], move, env->tmp);
            last_face = move / 3;
        }
    }
}

// ---------------------------------------------------------------------------
// Required interface functions
// ---------------------------------------------------------------------------
void init(Cube* env) {
    env->num_agents = 1;
    env->pool = (uint8_t (*)[CUBE_STICKERS])calloc(SCRAMBLE_POOL_SIZE, sizeof(env->pool[0]));
    build_pool(env);
}

void c_reset(Cube* env) {
    int idx = rand_r(&env->rng) % SCRAMBLE_POOL_SIZE;
    memcpy(env->cube, env->pool[idx], CUBE_STICKERS);
    int extra = rand_r(&env->rng) % (EXTRA_MOVES + 1);
    int last_face = -1;
    for (int i = 0; i < extra; i++) {
        int move;
        do { move = rand_r(&env->rng) % CUBE_MOVES; } while (move/3 == last_face);
        apply_move(env->cube, move, env->tmp);
        last_face = move / 3;
    }
    env->step_count = 0;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    update_obs(env);
}

void c_step(Cube* env) {
    float a = env->actions[0];
    int action = (a >= 0.0f && a < (float)CUBE_MOVES) ? (int)a : 0;
    if (action >= 0 && action < CUBE_MOVES)
        apply_move(env->cube, action, env->tmp);
    env->step_count++;

    int solved    = is_solved(env->cube);
    int truncated = !solved && (env->step_count >= env->max_steps);
    int done      = solved || truncated;

    if (done) {
        float reward = 0.0f;
        int correct_count = count_correct(env->cube);
        if (solved) {
            float log_steps = logf(1.0f + (float)env->step_count);
            float log_max   = logf(1.0f + (float)env->max_steps);
            reward = 1.0f - log_steps / log_max;
        } else {
            reward = -1.0f + correct_count / 54.0f;
        }
        env->rewards[0]   = reward;
        env->terminals[0] = 1.0f;

        env->log.episode_return      += reward;
        env->log.episode_length      += (float)env->step_count;
        env->log.solve_rate          += solved ? 1.0f : 0.0f;
        env->log.correct_at_terminal += (float)correct_count / 54.0f;
        env->log.n                   += 1.0f;

        c_reset(env);
    } else {
        env->rewards[0]   = 0.0f;
        env->terminals[0] = 0.0f;
        update_obs(env);
    }
}

void c_close(Cube* env) {
    if (env->pool) {
        free(env->pool);
        env->pool = NULL;
    }
}

// ---------------------------------------------------------------------------
// Raylib render (2D net view)
// ---------------------------------------------------------------------------
static const Color CUBE_FACE_COLORS[6] = {
    {255,255,255,255}, // 0=U White
    {255,255,  0,255}, // 1=D Yellow
    {  0,200,  0,255}, // 2=F Green
    {  0,  0,200,255}, // 3=B Blue
    {255,140,  0,255}, // 4=L Orange
    {200,  0,  0,255}, // 5=R Red
};

void c_render(Cube* env) {
    if (!IsWindowReady()) {
        InitWindow(600, 500, "PufferLib Cube");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) { CloseWindow(); exit(0); }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // 2D net cross layout (grid col, grid row):
    // face order: U=0,D=1,F=2,B=3,L=4,R=5
    static const int face_cols[6] = {1, 1, 1, 3, 0, 2};
    static const int face_rows[6] = {0, 2, 1, 1, 1, 1};

    const int sz = 44, gap = 2;
    const int ox = 50, oy = 50;
    const int stride = 3*sz + 3*gap + 8;

    for (int f = 0; f < CUBE_FACES; f++) {
        int bx = ox + face_cols[f] * stride;
        int by = oy + face_rows[f] * stride;
        for (int i = 0; i < CUBE_FACE_SIZE; i++) {
            int col = i % 3, row = i / 3;
            DrawRectangle(bx + col*(sz+gap), by + row*(sz+gap),
                          sz, sz, CUBE_FACE_COLORS[env->cube[f*9+i]]);
        }
    }

    char buf[64];
    snprintf(buf, sizeof(buf), "Step %d / %d", env->step_count, env->max_steps);
    DrawText(buf, 10, 460, 20, (Color){241,241,241,255});
    EndDrawing();
}
