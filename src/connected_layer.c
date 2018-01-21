#include "connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/**
 * 全连接层
 * 输入:
 *     batch 输入图片数目
 *     inputs 每一张图片的元素个数
 *     outputs 输出元素个数
 *     activation 激活函数
 *     batch_normalize 是否进行BN操作
 *     adam
 * */
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;     // 层的类型

    l.inputs = inputs;      // 输入元素个数
    l.outputs = outputs;    // 输出元素个数
    l.batch=batch;          //
    l.batch_normalize = batch_normalize;
    l.h = 1;        //  全连接层的h 和 w都是1
    l.w = 1;        //
    l.c = inputs;   //  全连接层的c是输入元素个数
    l.out_h = 1;    //  全连接层的输出 h和w都是1
    l.out_w = 1;    //
    l.out_c = outputs;  // 全连接层的输出c是输出元素个数

    l.output = calloc(batch*outputs, sizeof(float));  // 全连接层的所有输出分配内存
    l.delta = calloc(batch*outputs, sizeof(float));   // 全连接层的敏感度图分配内存

    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));

    l.weights = calloc(outputs*inputs, sizeof(float)); // 权重和偏置分配内存
    l.biases = calloc(outputs, sizeof(float));

    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);  // 初始化权重
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i){   // 初始化偏置
        l.biases[i] = 0;
    }

    if(adam){   // adam优化算法 每配内存空间
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize){  // batch_normalize 这个也要看看
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
    if (adam) {
        l.m_gpu =       cuda_make_array(0, inputs*outputs);
        l.v_gpu =       cuda_make_array(0, inputs*outputs);
        l.bias_m_gpu =  cuda_make_array(0, outputs);
        l.bias_v_gpu =  cuda_make_array(0, outputs);
        l.scale_m_gpu = cuda_make_array(0, outputs);
        l.scale_v_gpu = cuda_make_array(0, outputs);
    }

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
#endif
    }
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}
/**
 * 更新全连接层权值
 * */
void update_connected_layer(layer l, update_args a)
{   // 计算出当前学习率
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    // 当前运动惯量
    float momentum = a.momentum;
    //
    float decay = a.decay;
    int batch = a.batch;
    // bias更新
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    // 乘以momentum
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }
    // weight_updates = weight_updates + alpha*weights 计算出需要更新的weight
    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    // 更新weights
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    // 对当前的weight_updates进行scal处理
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}
/**
 * 前向传播
 * */
void forward_connected_layer(layer l, network net)
{
    // 初始化全连接层的所有输出（包含所有batch）为0值
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;      // 图片个数
    int k = l.inputs;     // 输入元素个数
    int n = l.outputs;    // 输出元素个数
    float *a = net.input;     // a：全连接层的输入数据，维度为l.batch*l.inputs（包含整个batch的输入），可视作l.batch行，l.inputs列，每行就是一张输入图片
    float *b = l.weights;     // b：全连接层的所有权重，维度为l.outputs*l.inputs(见make_connected_layer())
    float *c = l.output;      // c：全连接层的所有输出（包含所有batch），维度为l.batch*l.outputs（包含整个batch的输出）
    // 根据维度匹配规则，显然需要对b进行转置，故而调用gemm_nt()函数，最终计算得到的c的维度为l.batch*l.outputs,
    // 全连接层的的输出很好计算，直接矩阵相承就可以了，所谓全连接，就是全连接层的输出与输入的每一个元素都有关联（当然是同一张图片内的，
    // 最中得到的c有l.batch行,l.outputs列，每行就是一张输入图片对应的输出）
    // m：a的行，值为l.batch，含义为全连接层接收的一个batch的图片张数
    // n：b'的列数，值为l.outputs，含义为全连接层对应单张输入图片的输出元素个数
    // k：a的列数，值为l.inputs，含义为全连接层单张输入图片元素个数 b'的行数
    // b要经过转置
    // shape(a) l.batch*l.inputs
    // shape(b) l.outputs*l.inputs
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);  //矩阵运算
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_connected_layer(layer l, network net)
{   // 计算出梯度
	// l.output: l.batch, l.outputs
	// l.delta: l.batch, l.outputs
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
    	// 后向传播 计算 bias的更新
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }
    // a：当前全连接层敏感度图，维度为l.batch*l.outputs
    // b：当前全连接层所有输入，维度为l.batch*l.inputs
    // c：当前全连接层权重更新值，维度为l.outputs*l.inputs（权重个数）
    // 由行列匹配规则可知，需要将a转置，故而调用gemm_tn()函数，转置a实际上是想把batch中所有图片的影响叠加。
    // 全连接层的权重更新值的计算也相对简单，简单的矩阵乘法即可完成：当前全连接层的敏感度图乘以当前层的输入即可得到当前全连接层的权重更新值，
    // （当前层的敏感度是误差函数对于加权输入的导数，所以再乘以对应输入值即可得到权重更新值）
    // m：a'的行，值为l.outputs，含义为每张图片输出的元素个数
    // n：b的列数，值为l.inputs，含义为每张输入图片的元素个数
    // k：a’的列数，值为l.batch，含义为一个batch中含有的图片张数
    // 最终得到的c维度为l.outputs*l.inputs，对应所有权重的更新值
    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;
    // a = y.activate(y)'
    // b = x
    // c = 1*a'b + c   =>>> c: shape: outputs, inputs
    // c = y*activate(y)'*x
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;
    // 一定注意此时的c等于net.delta，已经在network.c中的backward_network()函数中赋值为上一层的delta
    // a：当前全连接层敏感度图，维度为l.batch*l.outputs
    // b：当前层权重（连接当前层与上一层），维度为l.outputs*l.inputs
    // c：上一层敏感度图（包含整个batch），维度为l.batch*l.inputs
    // 由行列匹配规则可知，不需要转置。由全连接层敏感度图计算上一层的敏感度图也很简单，直接利用矩阵相乘，将当前层l.delta与当前层权重相乘就可以了，
    // 只需要注意要不要转置，拿捏好就可以，不需要像卷积层一样，需要对权重或者输入重排！
    // m：a的行，值为l.batch，含义为一个batch中含有的图片张数
    // n：b的列数，值为l.inputs，含义为每张输入图片的元素个数
    // k：a的列数，值为l.outputs，含义为每张图片输出的元素个数
    // 最终得到的c维度为l.bacth*l.inputs（包含所有batch）
    // 当前层的delta会影响前一层delta的计算, 乘以w类似与求和的那个公式
    // c = c + a*b
    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}


void denormalize_connected_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.outputs; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .000001);
        for(j = 0; j < l.inputs; ++j){
            l.weights[i*l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/**
 * 输出该层的统计参数
 * */
void statistics_connected_layer(layer l)
{
    if(l.batch_normalize){
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void push_connected_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void update_connected_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }else{
        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

        if(l.batch_normalize){
            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        }

        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
    }
}

void forward_connected_layer_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_connected_layer_gpu(layer l, network net)
{
    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = net.input_gpu;
    float * c = l.weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = net.delta_gpu;

    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
}
#endif
