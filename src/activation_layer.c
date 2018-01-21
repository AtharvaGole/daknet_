#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/**
 * 激活函数层
 * batch
 * inputs
 * activation  激活函数
 * */
layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    l.type = ACTIVE;  //神经网络层的类型

    l.inputs = inputs;  // 输入数据大小
    l.outputs = inputs; // 输出数据大小
    l.batch=batch; // batch数据数量
    // 分配 内存空间
    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));
    // 指针函数 前向 后向
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}
// 前向传播 激活函数 net 应该只是临时的变量
void forward_activation_layer(layer l, network net)
{   // 将input数据 复制给output
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    // 针对 output当中的每一个数据 执行激活函数
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
// 反向传播 激活函数 类似于softmax层
void backward_activation_layer(layer l, network net)
{   // 用于计算梯度 注意这里的delta应该从会面一层传入 肯定是这样字的
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    // 将该层的数据 赋值给网络 net.delta 应该保留了所有层的数据
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_activation_layer_gpu(layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
