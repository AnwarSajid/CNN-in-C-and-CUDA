#ifndef __debug_H__
#define __debug_H__

#include "layer.h"
#include "dataset.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void display_weights_matrices(struct nnlayer *headlayer);

    void display_qweights_matrices(struct nnlayer *headlayer);

    void display_quant_stepsizes(cnnlayer_t *headlayer);

    void display_cnn_layers(cnnlayer_t* headlayer);

    void display_dataVector(struct dataset* dataVector);

    void display_gradientMap(cnnlayer_t* headlayer);

#ifdef __cplusplus
}
#endif

#endif
