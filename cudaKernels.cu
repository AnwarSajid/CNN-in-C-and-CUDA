#include "cudaKernels.h"
#include "error_handling.h"

__global__ void compute_transfer_function(real_t* d_output, real_t* d_biases, int layer_type)
{
    int dstIdx = blockIdx.x * blockDim.x + threadIdx.x;
    real_t netInput = d_output[dstIdx] + d_biases[blockIdx.x];

    if (layer_type == 1)
        d_output[dstIdx] = 1.0/(1 + exp(-netInput));
    else if (layer_type == 2) 
        d_output[dstIdx] = (1.7159 * tanh(0.66666 * netInput));
    else if (layer_type == 3)
        d_output[dstIdx] = ((netInput >= 0) ? netInput : 0); 
}

__global__ void convolve_device_2D(real_t* d_output, real_t* d_input, real_t* d_kernel, int kerSize)
{
    int inpfmapSize = blockDim.x * blockDim.y;
    int inprSize = (int) sqrt((float)inpfmapSize);    
    int inpcSize = inprSize; 
    int kerrSize = (int) sqrt((float)kerSize);    
    int kercSize = kerrSize; 
    int outrSize = inprSize - 2 * (kerrSize/2);

    extern __shared__ real_t sh_mem[];
    real_t* sh_input = &sh_mem[0];
    real_t* sh_kernel = &sh_mem[inpfmapSize];
    
    //1. copy input from global memory to shared memory
    int cinpThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int inpThreadIdx = blockIdx.x * (blockDim.x * blockDim.y) + cinpThreadIdx; //(0, 1) * (32 * 32) 
    sh_input[cinpThreadIdx] = d_input[inpThreadIdx];

    //2. copy kernel from global memory to shared memory (only 5x5 per thread block)
    int currBlockstIdx = (blockIdx.y * gridDim.x + blockIdx.x) * kerSize;
    int currBlockendIdx = currBlockstIdx + kerSize;

    int kIdx = blockIdx.y * (gridDim.x * kerSize) + (blockIdx.x * kerSize) + threadIdx.y * blockDim.x + threadIdx.x;
    if (kIdx >= currBlockstIdx && kIdx < currBlockendIdx) 
        sh_kernel[kIdx % kerSize] = d_kernel[kIdx]; 

    __syncthreads();

    int crowIdx = threadIdx.y;
    int ccolIdx = threadIdx.x;

    if (crowIdx >= kerrSize/2 && crowIdx < inprSize - kerrSize/2)
    {
        if (ccolIdx >= kercSize/2 && ccolIdx < inpcSize - kercSize/2)
        {
            int kr = 0, kc = 0, kCtr = 0;
            real_t sum = 0.0;

            for (kr = -kerrSize/2; kr <= kerrSize/2; kr++)
            {
                for (kc = -kercSize/2; kc <= kercSize/2; kc++)
                {
                    sum += sh_input[cinpThreadIdx + kr * inprSize + kc] * sh_kernel[kCtr];
                    kCtr++; 
                }   
            } 

            __syncthreads();
            int dstIdx = blockIdx.y * (outrSize * outrSize) + (crowIdx - kerrSize/2) * outrSize + (ccolIdx - kercSize/2);
            atomicAdd(&d_output[dstIdx], sum);
        }
    }
}

__global__ void errorbp_convolution_update_biases(real_t* d_lerr_deltas, real_t* d_delta_biases)
{
    int gmIdx = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ real_t sum_err_deltas;

    if ((threadIdx.y * blockDim.x + threadIdx.x) == 0)
        sum_err_deltas = 0.0;

    __syncthreads(); //Thread zero may be late, so sync needed I think

    atomicAdd(&sum_err_deltas, d_lerr_deltas[gmIdx]);

    __syncthreads(); 
    d_delta_biases[blockIdx.x] = sum_err_deltas;    
}

__global__ void errorbp_convolution_layers2(real_t* d_output, real_t* d_lerr_deltas, real_t* d_cerr_deltas, real_t* d_weights, real_t* d_delta_weights, int kerSize)
{
    int inpfmapSize = blockDim.x * blockDim.y;
    int inprSize = (int) sqrt((float)inpfmapSize);    
    int inpcSize = inprSize; 
    int kerrSize = (int) sqrt((float)kerSize);    
    int kercSize = kerrSize; 
    int prevlrSize = inprSize - 2 * (kerrSize/2);
    int prevlcSize = inpcSize - 2 * (kercSize/2);

    /* Compute delta_weights and current layer error deltas */
    extern __shared__ real_t sh_mem[];

    real_t* sh_output = &sh_mem[0];
    real_t* sh_plerr_deltas = &sh_mem[blockDim.x * blockDim.y];
    real_t* sh_weights = &sh_mem[blockDim.x * blockDim.y + prevlrSize * prevlcSize];

    /*1. copy current layer outputs to sh_output */
    int cThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int outputIdx = blockIdx.x * blockDim.x * blockDim.y + cThreadIdx;
    sh_output[cThreadIdx] = d_output[outputIdx];

    /*2. copy last layer error deltas to shared memory sh_plerr_deltas */
    int plerrIdx = blockIdx.y * prevlrSize * prevlcSize + cThreadIdx;

    int plerrStIdx = blockIdx.y * prevlrSize * prevlcSize;
    int plerrEndIdx = plerrStIdx + prevlrSize * prevlcSize;

    if (plerrIdx >= plerrStIdx && plerrIdx < plerrEndIdx) 
        sh_plerr_deltas[cThreadIdx % (prevlrSize * prevlcSize)] = d_lerr_deltas[plerrIdx];
    
    /*3. copy d_weights to sh_weights (shared memory) */
    int currBlockstIdx = (blockIdx.y * gridDim.x * kerSize) + (blockIdx.x * kerSize);
    int currBlockendIdx = currBlockstIdx + kerSize;

    int kIdx = (blockIdx.y * gridDim.x * kerSize) + (blockIdx.x * kerSize) + threadIdx.y * blockDim.x + threadIdx.x;
    if (kIdx >= currBlockstIdx && kIdx < currBlockendIdx) 
        sh_weights[kIdx % kerSize] = d_weights[kIdx]; 

    __syncthreads();

    /* Compute delta weights */
    int crowIdx = threadIdx.y;
    int ccolIdx = threadIdx.x;
    
    if (crowIdx >= kerrSize/2 && crowIdx < inprSize - kerrSize/2)
    {
        if (ccolIdx >= kercSize/2 && ccolIdx < inpcSize - kercSize/2)
        {
            int cplIdx = (threadIdx.y - (kerrSize/2)) * prevlcSize + threadIdx.x - (kercSize/2);
            int plUnitIdx = cplIdx % (prevlrSize * prevlcSize); 
            int dwIdx = blockIdx.y * gridDim.x * kerSize + blockIdx.x * kerSize;
            
            int kr = 0, kc = 0, wCtr = 0;
            real_t dweight = 0.0, cerrdelta = 0;

            for (kr = -kerrSize/2; kr <= kerrSize/2; kr++)
            {
                for (kc = -kercSize/2; kc <= kercSize/2; kc++)
                {
                    dweight = sh_output[cThreadIdx + kr * inprSize + kc] * sh_plerr_deltas[plUnitIdx]; 
                    atomicAdd(&d_delta_weights[dwIdx + wCtr], dweight); 
                    
                    // compute error deltas for current layer
                    cerrdelta = sh_weights[wCtr] * sh_plerr_deltas[plUnitIdx]; 
                    atomicAdd(&d_cerr_deltas[outputIdx + kr * inprSize + kc], cerrdelta);
                    wCtr++;
                }
            }
        }
    }
}

void copy_dweights_to_hweights(struct nnlayer* headlayer)
{
    struct nnlayer* current = headlayer;
    struct nnlayer* next_to_current = current->next;

    bool_t flag = true;
    while (current != NULL && flag == true)
    {
        if (next_to_current->next == NULL)
            flag = false;

        int ndweights = 0;
        if (next_to_current->subsampling == false)
            ndweights = current->no_of_fmaps * next_to_current->no_of_fmaps * current->fkernel * current->fkernel;
        else
            ndweights = current->no_of_fmaps;

        int wSize = ndweights * sizeof(real_t);
        HANDLE_ERROR(cudaMemcpy(current->weights_matrix, current->d_weights, wSize, cudaMemcpyDeviceToHost));

        // biases 
        int bSize = next_to_current->no_of_fmaps * sizeof(real_t);
        HANDLE_ERROR(cudaMemcpy(current->biases, current->d_biases, bSize, cudaMemcpyDeviceToHost));

        if (flag == true)
        {
            current = next_to_current;
            next_to_current = current->next;
        }
    }
}

__global__ void errorbp_convolution_layers(real_t* d_output, real_t* d_lerr_deltas, real_t* d_cerr_deltas, real_t* d_weights, real_t* d_delta_weights, int kerSize)
{
    int inpfmapSize = blockDim.x * blockDim.y;
    int inprSize = (int) sqrt((float)inpfmapSize);    
    int inpcSize = inprSize; 
    int kerrSize = (int) sqrt((float)kerSize);    
    int kercSize = kerrSize; 
    int prevlrSize = inprSize - 2 * (kerrSize/2);
    int prevlcSize = inpcSize - 2 * (kercSize/2);

    /* Compute delta_weights and current layer error deltas */
    extern __shared__ real_t sh_mem[];

    real_t* sh_output = &sh_mem[0];
    real_t* sh_plerr_deltas = &sh_mem[blockDim.x * blockDim.y];
    real_t* sh_weights = &sh_mem[blockDim.x * blockDim.y + prevlrSize * prevlcSize];

    /*1. copy current layer outputs to sh_output */
    int cThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int outputIdx = blockIdx.x * blockDim.x * blockDim.y + cThreadIdx;
    sh_output[cThreadIdx] = d_output[outputIdx];

    /*2. copy last layer error deltas to shared memory sh_plerr_deltas */
    int plerrIdx = blockIdx.y * prevlrSize * prevlcSize + cThreadIdx;
    //if (plerrIdx <= prevlrSize * prevlcSize) //This statement is causing problem !!!!!!!!
      //  sh_plerr_deltas[cThreadIdx] = d_lerr_deltas[plerrIdx]; //both indices are wrong I think
   
    int plerrStIdx = blockIdx.y * prevlrSize * prevlcSize;
    int plerrEndIdx = plerrStIdx + prevlrSize * prevlcSize;
    if (plerrIdx >= plerrStIdx && plerrIdx < plerrEndIdx)
        sh_plerr_deltas[cThreadIdx % (prevlrSize * prevlcSize)] = d_lerr_deltas[plerrIdx];
    
    /*3. copy d_weights to sh_weights (shared memory) */
    int currBlockstIdx = (blockIdx.y * gridDim.x * kerSize) + (blockIdx.x * kerSize);
    int currBlockendIdx = currBlockstIdx + kerSize;

    int kIdx = (blockIdx.y * gridDim.x * kerSize) + (blockIdx.x * kerSize) + threadIdx.y * blockDim.x + threadIdx.x;
    if (kIdx >= currBlockstIdx && kIdx < currBlockendIdx) 
        sh_weights[kIdx % kerSize] = d_weights[kIdx]; 

    __syncthreads();

    /* Compute delta weights */
    int crowIdx = threadIdx.y;
    int ccolIdx = threadIdx.x;

    if (crowIdx >= kerrSize/2 && crowIdx < inprSize - kerrSize/2)
    {
        if (ccolIdx >= kercSize/2 && ccolIdx < inpcSize - kercSize/2)
        {
            int plUnitIdx = cThreadIdx % (prevlrSize * prevlcSize);
            int dwIdx = blockIdx.y * gridDim.x * kerSize + blockIdx.x * kerSize;
              
            int kr = 0, kc = 0, wCtr = 0;
            real_t dweight = 0.0, cerrdelta = 0;

            for (kr = -kerrSize/2; kr <= kerrSize/2; kr++)
            {
                for (kc = -kercSize/2; kc <= kercSize/2; kc++)
                {
                    dweight = sh_output[cThreadIdx + kr * inprSize + kc] * sh_plerr_deltas[plUnitIdx]; 
                    atomicAdd(&d_delta_weights[dwIdx + wCtr], dweight); 
                    
                    // compute error deltas for current layer
                    cerrdelta = sh_weights[wCtr] * sh_plerr_deltas[plUnitIdx]; 
                    atomicAdd(&d_cerr_deltas[outputIdx + kr * inprSize + kc], cerrdelta);
                    wCtr++;
                }
            }
        }
    }    
}

__global__ void d_errbp_subsampling(real_t* d_output, real_t* d_lerr_deltas, real_t* d_cerr_deltas, int* d_gradientMap, int layer_type)
{
    extern __shared__ real_t sh_mem[];
    real_t* sh_lerr_deltas = &sh_mem[0];
    real_t* sh_gradientMap = &sh_mem[blockDim.x * blockDim.y];
   
    /* 1. copy last layer error deltas to shared memory */
    int cThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int  plUnitIdx = blockIdx.x * (blockDim.x * blockDim.y) + cThreadIdx;
    sh_lerr_deltas[cThreadIdx] = d_lerr_deltas[plUnitIdx];

    /* 2. copy gradientMap to shared memory */
    sh_gradientMap[cThreadIdx] = d_gradientMap[plUnitIdx];

    __syncthreads();

    /* 3. compute current error deltas */
    int gMIdx = sh_gradientMap[cThreadIdx];
    real_t y = d_output[gMIdx];

    real_t dy = 0; 
    if (layer_type == 1)
    {
        dy = y * (1 - y); 
    }
    else if (layer_type == 2)
    {
        real_t A = 1.7159;
        real_t S = 2.0/3.0;
        dy = A * S * (1 - y/A) * (1 + y/A); 
    }
    else if (layer_type == 3)
    {
        if (y > 0) dy = 1;
        else dy = 0;      
    }

    d_cerr_deltas[gMIdx] = sh_lerr_deltas[cThreadIdx] * dy;
}

__global__ void d_subsampling(real_t* d_output, real_t* d_input, int* d_gradientMap, int layer_type)
{
    extern __shared__ real_t sh_mem[];
    real_t* sh_input = &sh_mem[0];
    
    //1. current Thread Index
    int cThreadIdx = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int cinpIdx = threadIdx.y * blockDim.x + threadIdx.x;
    sh_input[cinpIdx] = d_input[cThreadIdx];

    __syncthreads();

    // This one is currently only MAX Pooling
    if (threadIdx.x % 2 == 0 && threadIdx.y % 2 == 0)
    {
        real_t n1 = sh_input[cinpIdx];
        real_t n2 = sh_input[cinpIdx + 1];
        real_t n3 = sh_input[cinpIdx + blockDim.x];
        real_t n4 = sh_input[cinpIdx + blockDim.x + 1];
        real_t max = 0.0;

        int dstThreadIdx = blockIdx.x * (blockDim.x/2) * (blockDim.y/2) + (threadIdx.y/2) * (blockDim.x/2) + threadIdx.x/2;

        if (n1 >= n2 && n1 >= n3 && n1 >= n4)
        {
            max = n1;
            d_gradientMap[dstThreadIdx] = cThreadIdx; 
        }
        else if (n2 >= n1 && n2 >= n3 && n2 >= n4)
        {
            max = n2;
            d_gradientMap[dstThreadIdx] = cThreadIdx + 1;
        }
        else if (n3 >= n1 && n3 >= n2 && n3 >= n4)
        {
            max = n3;
            d_gradientMap[dstThreadIdx] = cThreadIdx + blockDim.x;
        }
        else if (n4 >= n1 && n4 >= n2 && n4 >= n3)
        {
            max = n4;
            d_gradientMap[dstThreadIdx] = cThreadIdx + blockDim.x + 1;
        }

        real_t fmax = 0;
        if (layer_type == 1)
        {
            fmax = 1.0/(1 + exp(-max));
        }
        else if (layer_type == 2)
        {
            fmax = 1.7159 * tanh(0.66666 * max);
        }
        else if (layer_type == 3)
        {
            if (max >= 0)
                fmax = max;
            else
                fmax = 0;            
        }

        d_output[dstThreadIdx] = fmax;
    }
}

__global__ void d_rear_DNN(real_t* d_output, real_t* d_input, real_t* d_weights)
{
    extern __shared__ real_t sh_mem[];
    real_t* sh_input = &sh_mem[0];
    real_t* sh_weights = &sh_mem[blockDim.x];
    
    //1. current Weight, input Thread Index
    sh_input[threadIdx.x] = d_input[threadIdx.x];

    int cwThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    sh_weights[cwThreadIdx % blockDim.x] = d_weights[cwThreadIdx];
     
    __syncthreads();

    //2. getting product of w*x
    real_t wx = sh_weights[threadIdx.x] * sh_input[threadIdx.x];
    atomicAdd(&d_output[blockIdx.x], wx); 
}

__global__ void d_update_weights_kernel2(real_t* d_weights, real_t* d_delta_weights, double d_LEARNING_RATE)
{
    int cIdx = threadIdx.y * blockDim.x + threadIdx.x;
    d_weights[cIdx] = d_weights[cIdx] - d_LEARNING_RATE * d_delta_weights[cIdx];
}

__global__ void d_update_weights_kernel(real_t* d_weights, real_t* d_delta_weights, double d_LEARNING_RATE)
{
    extern __shared__ real_t sh_mem[];
    real_t* sh_weights = &sh_mem[0];
    real_t* sh_delta_weights = &sh_mem[blockDim.x * blockDim.y];

    int cThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;

    /*1. copy weights to shared memory */
    int cIdx = blockIdx.y * gridDim.x * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y + cThreadIdx;
    sh_weights[cThreadIdx] = d_weights[cIdx];

    /*2. copy delta_weights to shared memory */
    sh_delta_weights[cThreadIdx] = d_delta_weights[cIdx];
    
    __syncthreads();

    d_weights[cIdx] = sh_weights[cThreadIdx] - d_LEARNING_RATE * sh_delta_weights[cThreadIdx];
}

__global__ void d_update_biases_kernel(real_t* d_biases, real_t* d_delta_biases, double d_LEARNING_RATE)
{
    d_biases[blockIdx.x] = d_biases[blockIdx.x] - d_LEARNING_RATE * d_delta_biases[blockIdx.x];
}


// Error Back Propagation (The rear classifier layers) 
__global__ void d_rear_DNN_update_error_deltas(real_t* d_output, real_t* d_cerr_deltas, int layer_type)
{
    extern __shared__ real_t sh_mem[];
    real_t* sh_output = &sh_mem[0];
    sh_output[threadIdx.x] = d_output[threadIdx.x];

    __syncthreads();

    real_t cedelta = d_cerr_deltas[threadIdx.x];

    /* Multiply error deltas with gradient*/
    real_t y = sh_output[threadIdx.x];
    if (layer_type == 1)
    {
        real_t dy = y * (1 - y);
        d_cerr_deltas[threadIdx.x] = cedelta * dy;
    }
    else if (layer_type == 2)
    {
        real_t A = 1.7159;
        real_t S = 2.0/3.0;
        real_t dy = A * S * (1 - y/A) * (1 + y/A); 

        d_cerr_deltas[threadIdx.x] = cedelta * dy; 
    }
    else if (layer_type == 3)
    {
        real_t dy = 0;
        if (y > 0) dy = 1;
        else dy = 0;

        d_cerr_deltas[threadIdx.x] = cedelta * dy; 
    }
}

__global__ void d_update_error_deltas(real_t* d_output, real_t* d_cerr_deltas, int layer_type)
{
    extern __shared__ real_t sh_mem[];
    real_t* sh_output = &sh_mem[0];

    int cThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    sh_output[threadIdx.x] = d_output[cThreadIdx];

    __syncthreads();

    real_t cedelta = d_cerr_deltas[cThreadIdx];

    /* Multiply error deltas with gradient*/
    real_t y = sh_output[threadIdx.x];
    if (layer_type == 1)
    {
        real_t dy = y * (1 - y);
        d_cerr_deltas[cThreadIdx] = cedelta * dy; 
    }
    else if (layer_type == 2)
    {
        real_t A = 1.7159;
        real_t S = 2.0/3.0;
        real_t dy = A * S * (1 - y/A) * (1 + y/A); 

        d_cerr_deltas[cThreadIdx] = cedelta * dy; 
    }
    else if (layer_type == 3)
    {
        real_t dy = 0;
        if (y > 0) dy = 1;
        else dy = 0;

        d_cerr_deltas[cThreadIdx] = cedelta * dy; 
    }
}

__global__ void d_rear_DNN_errorbp(real_t* d_output, real_t* d_lerr_deltas, real_t* d_cerr_deltas, real_t* d_weights,real_t* d_delta_weights,real_t* d_delta_biases)
{
    extern __shared__ real_t sh_mem[];
    real_t* sh_output = &sh_mem[0];
    real_t* sh_weights = &sh_mem[blockDim.x];
    real_t* sh_lerr_delta = &sh_mem[2 * blockDim.x];

    /* Copy d_output from global memory to shared memory (Pushing the gradients backwards)*/
    sh_output[threadIdx.x] = d_output[threadIdx.x];

    /* Copy weights to shared memory */
    int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
    sh_weights[threadIdx.x] = d_weights[cIdx]; 

    /* Copy last layer error delta to shared memory */
    sh_lerr_delta[0] = d_lerr_deltas[blockIdx.x];

    __syncthreads();

    /* compute delta weights (change in weights) */
    d_delta_weights[cIdx] =  sh_lerr_delta[0] * sh_output[threadIdx.x];
    d_delta_biases[blockIdx.x] =  sh_lerr_delta[0] * 1.0; 

    /* accumulate error gradients */
    atomicAdd(&d_cerr_deltas[threadIdx.x], d_lerr_deltas[blockIdx.x] * sh_weights[threadIdx.x]); 
}

void copy_hweights_to_dweights(struct nnlayer* headlayer)
{
    struct nnlayer* current = headlayer;
    struct nnlayer* next_to_current = current->next;

    bool_t flag = true;
    while (current != NULL && flag == true)
    {
        if (next_to_current->next == NULL)
            flag = false;

        int ndweights = 0;
        if (next_to_current->subsampling == false)
            ndweights = current->no_of_fmaps * next_to_current->no_of_fmaps * current->fkernel * current->fkernel;
        else
            ndweights = current->no_of_fmaps;

        int wSize = ndweights * sizeof(real_t);
        HANDLE_ERROR(cudaMemcpy(current->d_weights, current->weights_matrix, wSize, cudaMemcpyHostToDevice));

        // biases 
        int bSize = next_to_current->no_of_fmaps * sizeof(real_t);
        HANDLE_ERROR(cudaMemcpy(current->d_biases, current->biases, bSize, cudaMemcpyHostToDevice));


        if (flag == true)
        {
            current = next_to_current;
            next_to_current = current->next;
        }
    }
}
