#include "cost_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
/*
** 支持四种cost functions
** L1    即误差的绝对值之和
** SSE   即L2，误差的平方和（可以查看blas.c中的l2_cpu()函数，没有乘以1/2），为绝大部分网络采用,
         全称应该是the sum of squares due to error（参考的：http://blog.sina.com.cn/s/blog_628033fa0100kjjy.html）
** MASKED 目前只发现在darknet9000.cfg中使用
** SMOOTH
*/

COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, "seg")==0) return SEG;
    if (strcmp(s, "sse")==0) return SSE;
    if (strcmp(s, "masked")==0) return MASKED;
    if (strcmp(s, "smooth")==0) return SMOOTH;
    if (strcmp(s, "L1")==0) return L1;
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}
/**
 * 对应不同的cost
 * */
char *get_cost_string(COST_TYPE a)
{
    switch(a){
        case SEG:
            return "seg";
        case SSE:
            return "sse";
        case MASKED:
            return "masked";
        case SMOOTH:
            return "smooth";
        case L1:
            return "L1";
    }
    return "sse";
}
/**
 * 创建cost层
 * 输入:
 *     batch  输入图片数目
 *     inputs 每一副图片的像素数目
 *     cost_type 代价函数类型
 *     scale 图像变换的一个参数
 * */
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
    fprintf(stderr, "cost                                           %4d\n",  inputs);
    cost_layer l = {0};
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    // 分配内存空间
    l.delta = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_cost_layer;
    l.backward = backward_cost_layer;
    #ifdef GPU
    l.forward_gpu = forward_cost_layer_gpu;
    l.backward_gpu = backward_cost_layer_gpu;

    l.delta_gpu = cuda_make_array(l.output, inputs*batch);
    l.output_gpu = cuda_make_array(l.delta, inputs*batch);
    #endif
    return l;
}

void resize_cost_layer(cost_layer *l, int inputs)
{
    l->inputs = inputs;
    l->outputs = inputs;
    // 是否需要重新分配内存空间
    l->delta = realloc(l->delta, inputs*l->batch*sizeof(float));
    l->output = realloc(l->output, inputs*l->batch*sizeof(float));
#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
    l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
#endif
}
/**
 * 前向传播 计算好cost层的delta
 * */
void forward_cost_layer(cost_layer l, network net)
{
    if (!net.truth) return;
    if(l.cost_type == MASKED){
        int i;
        for(i = 0; i < l.batch*l.inputs; ++i){
            if(net.truth[i] == SECRET_NUM) net.input[i] = SECRET_NUM;
        }
    }
    // faster rcnn当中用到这个代价函数
    if(l.cost_type == SMOOTH){
        smooth_l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }else if(l.cost_type == L1){
    	//l1 类型
        l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    } else {
    	// l.delta就是这样有数据的
    	//l2 类型
        l2_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }
    // 计算当次batch当中所有的输出数目和
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}
/**
 * 后向传播 将delta送给前一层(如:softmax)
 * */
void backward_cost_layer(const cost_layer l, network net)
{
	// net.delta += l.scala*l.delta
	axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_cost_layer(cost_layer l)
{
    cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void push_cost_layer(cost_layer l)
{
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

int float_abs_compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    if(fa < 0) fa = -fa;
    float fb = *(const float*) b;
    if(fb < 0) fb = -fb;
    return (fa > fb) - (fa < fb);
}

void forward_cost_layer_gpu(cost_layer l, network net)
{
    if (!net.truth_gpu) return;
    if(l.smooth){
        scal_gpu(l.batch*l.inputs, (1-l.smooth), net.truth_gpu, 1);
        add_gpu(l.batch*l.inputs, l.smooth * 1./l.inputs, net.truth_gpu, 1);
    }
    if (l.cost_type == MASKED) {
        mask_gpu(l.batch*l.inputs, net.input_gpu, SECRET_NUM, net.truth_gpu);
    }

    if(l.cost_type == SMOOTH){
        smooth_l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == L1){
        l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else {
        l2_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    }

    if (l.cost_type == SEG && l.noobject_scale != 1) {
        scale_mask_gpu(l.batch*l.inputs, l.delta_gpu, 0, net.truth_gpu, l.noobject_scale);
        scale_mask_gpu(l.batch*l.inputs, l.output_gpu, 0, net.truth_gpu, l.noobject_scale);
    }

    if(l.ratio){
        cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
        qsort(l.delta, l.batch*l.inputs, sizeof(float), float_abs_compare);
        int n = (1-l.ratio) * l.batch*l.inputs;
        float thresh = l.delta[n];
        thresh = 0;
        printf("%f\n", thresh);
        supp_gpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
    }

    if(l.thresh){
        supp_gpu(l.batch*l.inputs, l.thresh*1./l.inputs, l.delta_gpu, 1);
    }

    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}

void backward_cost_layer_gpu(const cost_layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

