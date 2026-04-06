#include "cube.h"

#define OBS_SIZE    54
#define NUM_ATNS     1
#define ACT_SIZES   {18}
#define OBS_TENSOR_T FloatTensor

#define Env Cube
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_return",      log->episode_return);
    dict_set(out, "episode_length",      log->episode_length);
    dict_set(out, "solve_rate",          log->solve_rate);
    dict_set(out, "correct_at_terminal", log->correct_at_terminal);
    dict_set(out, "n",                   log->n);
}
