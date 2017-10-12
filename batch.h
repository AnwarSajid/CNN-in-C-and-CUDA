#ifndef __BATCH_H__
#define __BATCH_H__

#define BATCH_SIZE 1

#ifdef __cplusplus
extern "C"
{
#endif

#include "layer.h"

void mini_batching(int* batch_indexes, int max_index, bool_t shuffle);

#ifdef __cplusplus
}
#endif

#endif 
