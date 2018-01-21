#include "blas.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}
/**
 * 某层神经网络展开,一般用于卷积层输出和全连接层输入部分
 * x 包含所有数据，比如l.output，其包含的元素个数为l.batch*l.outputs
 * size 所在层神经元数目
 * layers 神经网络的层数
 * batch 一个batch中包含的图片张数，即l.batch
 * forward 是否是前向传播
 * */
void flatten(float *x, int size, int layers, int batch, int forward)
{   // 分配内存地址 临时存储数据
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){   // 遍历每一幅图片
        for(c = 0; c < layers; ++c){  // 遍历每一层
            for(i = 0; i < size; ++i){  // 遍历每一层当中的每一个神经元
                int i1 = b*layers*size + c*size + i;    // 第b幅图像在第c层的第i个神经元对应的下标
                int i2 = b*layers*size + i*layers + c;  // 对应的新下标
                if (forward) swap[i2] = x[i1]; // 前向传播
                else swap[i1] = x[i2]; // 反向传播
            }
        }
    }
    // 复制计算得到的新的数据
    memcpy(x, swap, size*layers*batch*sizeof(float));
    // 释放临时计算的内存空间
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}

/*
** 有组织的计算输入数据x的平均值，输出的mean是一个矢量，比如如果x是多张3通道的图片，那么mean的维度就为通道数3
** （也即每张输入图片会得到3张特征图）,为方便，我们称这三个通道分别为第一，第二，第三通道，由于每次训练输入的都是一个batch的图片，
** 因此最终会输出batch张三通道的图片，mean中的第一个元素就是第一个通道上全部batch张输出特征图所有元素的平均值，依次类推
** 本函数的主要用处之一应该就是实现batch normalization的第一步了！
** 输入：
**       x         包含所有数据，比如l.output，其包含的元素个数为l.batch*l.outputs
**       batch     一个batch中包含的图片张数，即l.batch
**       filters   该层神经网络的滤波器个数，也即该层网络输出图片的通道数（比如对卷积网络来说，就是核的个数了）
**       spatial   该层神经网络每张输出特征图的尺寸，也即等于l.out_w*l.out_h
**       mean      求得的平均值，维度为filters，也即每个滤波器对应有一个均值（每个滤波器会处理所有图片）
** 说明： 该函数的具体调用可以参考：batchnorm_layer.c中的forward_batchnorm_layer()函数
** 说明2：mean_cpu()函数是一个纯粹的数学计算函数，有组织的计算x中某些数据的均值，x的具体存储结构视具体情况而定，
**       在写注释时，主要参考了batchnorm_layer.c中的forward_batchnorm_layer()函数对该函数的调用，
**       因此有些注释就加上了一些具体含义，结合这些含义会有助于理解，但还是要记住，这是一个一般的数学计算函数，
**       不同地方调用该函数可能有不同的含义。
** 说明3：均值是哪些数据的均值？x中包含了众多数据，mean中的每个元素究竟对应x中哪些数据的平均值呢？
**       此处还是结合batchnorm_layer.c中的forward_batchnorm_layer()函数的调用来解释，
**       其中的x为l.output，有l.batch行，每行有l.out_c*l.out_w*l.out_h个元素，每一行又可以分成
**       l.out_c行，l.out_w*l.out_h列，那么l.mean中的每一个元素，是某一个通道上所有batch的输出的平均值
**       （比如卷积层，有3个核，那么输出通道有3个，每张输入图片都会输出3张特征图，可以理解每张输出图片是3通道的，
**       若每次输入batch=64张图片，那么将会输出64张3通道的图片，而mean中的每个元素就是某个通道上所有64张图片
**       所有元素的平均值，比如第1个通道上，所有64张图片像素平均值）
** 说明4：在全连接层的前向传播函数中：sptial=1，因为全连接层的输出可以看作是1*1的特征图
*/
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
	// scale是求均值中的分母部分
    float scale = 1./(batch * spatial);
    int i,j,k;
    // 遍历不同的filter
    for(i = 0; i < filters; ++i){
        mean[i] = 0; // 初始化值0
        for(j = 0; j < batch; ++j){  // 遍历每一个filter里面的所有图像
            for(k = 0; k < spatial; ++k){
            	// 计算出偏移位置
            	//  j*filters*spatial 是第j个图像的位置偏移量
            	//  i*spatial 是第i个filter 位置偏移量
            	//  k是第k个元素的偏移位置
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index]; // 求和
            }
        }
        mean[i] *= scale; // 求出平均值
    }
}
/*
** 计算输入x中每个元素的方差（大致的过程和上面的mean_cpu类似，不再赘述）
** 本函数的主要用处之一应该就是batch normalization的第二步了！
** 输入：
**       x         包含所有数据，比如l.output，其包含的元素个数为l.batch*l.outputs
**       batch     一个batch中包含的图片张数，即l.batch
**       filters   该层神经网络的滤波器个数，也即该层网络输出图片的通道数（比如对卷积网络来说，就是核的个数了）
**       spatial   该层神经网络每张输出特征图的尺寸，也即等于l.out_w*l.out_h
**       mean      求得的平均值，维度为filters，也即每个滤波器对应有一个均值（每个滤波器会处理所有图片）
*/
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{	// 计算方差时候的分母部分
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
            	// 计算出偏移位置
            	//  j*filters*spatial 是第j个图像的位置偏移量
            	//  i*spatial 是第i个filter 位置偏移量
            	//  k是第k个元素的偏移位置
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);  // 计算数据的平方
            }
        }
        variance[i] *= scale;  // 除以分母
    }
}
/**
 * 数据的标准化处理
 * 输入:
 *        x         包含所有数据，比如l.output，其包含的元素个数为l.batch*l.outputs
 *        mean      求得的平均值，维度为filters，也即每个滤波器对应有一个均值（每个滤波器会处理所有图片）
 *        variance  求得的方差值, 维度为filters，也即每个滤波器对应有一个方差（每个滤波器会处理所有图片）度为
 * */
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
            	// 计算出偏移位置
            	//  j*filters*spatial 是第j个图像的位置偏移量
            	//  i*spatial 是第i个filter 位置偏移量
            	//  k是第k个元素的偏移位置
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}
// 赋值为常数
void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}
// 乘法
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}
// 平方
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}
// 线性变换 y = alpha*x + y
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}
// x = alpha*x
void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}
// 插值
void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}
// 复制
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}
// z = z + x*y
void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}
/**
 * 一种损失函数的计算方法 目标是什么 暂时不清楚 没有用过
 * 输入:
 *     n 类别数目
 *     pred 预测的数值
 *     truth 真实的数值
 * 输出:
 *     delta 导数
 *     error 误差值
 * **/

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){               // 遍历每一个类别
        float diff = truth[i] - pred[i];  // 计算误差 [0~1]
        float abs_val = fabs(diff);       // 计算误差的绝对值
        if(abs_val < 1) {                 // 误差较小
            error[i] = diff * diff;
            delta[i] = diff;              // 计算求出的导数 具体要看论文喽..
        }
        else {                            // 误差>=1的情况
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}
/**
 * 计算l1数值
 * 输出:
 *     delta 导数    delta = diff > 0 ? 1 : -1
 *     error 误差值  diff = |truth-pred|
 * */
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}
/**
 * 计算l2数值  平方误差
 * 输出:
 *     delta 导数    (truth - pred)
 *     error 误差值  (truth - pred)*(truth - pred)
 * */
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}
/**
 * 两个矩阵的dot后数值
 * */
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}
/*
** 输入： input   一组输入图片数据（含义见下面softmax_cpu()注释，下同）
**       n       一组输入数据中含有的元素个数n=l.inputs/l.groups
**       temp    温度参数，关于softmax的温度参数，可以搜索一下softmax with temperature，应该会有很多的
**       stride  跨度
**       output  这一组输入图片数据对应的输出（也即l.output中与这一组输入对应的某一部分）
** 说明：本函数实现的就是标准的softmax函数处理，唯一有点变化的就是在做指数运算之前，将每个输入元素减去了该组输入元素中的最大值，以增加数值稳定性，
**      关于此，可以参考博客：http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/，
**      这篇博客写的不错，博客中还提到了softmax-loss，此处没有实现（此处实现的也即博客中提到的softmax函数，将softmax-loss分开实现了）。
*/
void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    // 赋初始最大值为float中的最小值-FLT_MAX（定义在float.h中）
    float largest = -FLT_MAX;
    // 寻找输入中的最大值，至于为什么要找出最大值，是为了数值计算上的稳定，详细请戳：http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
    // 这篇博客写的不错，博客在接近尾声的时候，提到了为什么要减去输入中的最大值。
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride]; // 找出最大值
    }
    for(i = 0; i < n; ++i){
        // 在进行指数运算之间，如上面博客所说，首先减去最大值（当然温度参数也要除）
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;  // 求和
        output[i*stride] = e;   // 并将每一个输入的结果保存在相应的输出中
    }
    // 最后一步：归一化转换为概率（就是softmax函数的原型～），最后的输出结果保存在output中
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

/*
** 输入： input    softmax层所有输入数据（包含整个batch的），即net.input（上一层的输出）
**       n        一组输入数据中含有的元素个数n=l.inputs/l.groups
**       batch    一个batch中所含有的图片张数（等于net.batch）
**       batch_offset    一张输入图片含有的元素个数，即值等于l.inputs（所以叫做batch_offset，目的是要借助该参数在input中整张整张照片移位）
**       groups   一张输入图片的元素被分成了几组，值为l.groups（这个参数由配置文件指定，如果未指定，则默认为1）,这个参数暂时还没遇到怎么用，大部分的网络值都为1,也即相当于没有这个参数
**       group_offset    值等于n，组偏移（在每张输入图片元素中整组整组偏移）
**       stride   跨度，这个参数类似于axpy_cpu()函数中的INCX参数，一定注意不同于卷积层中的l.stride，这个参数是指按照stride间隔从每组输入数据中抽取元素，即会抽取所有索引为stride倍数的输入元素，
**                而其他的输入元素，实际没有用到；stride=1时，显然，相当于没有这个参数，所有输入数据都用到了（这个参数在softmax_layer层中，相当于没用，因为在forward_softmax_layer()中，
**                调用该函数时，stride已经被写死为1,并不能改，不知道还有没有其他地方使用了这个参数）
**       temp     softmax的温度参数l.temperature，关于softmax的温度参数，可以搜索一下softmax with temperature，应该会有很多的
**       output   经softmax处理之后得到的输出l.output（即概率），与input具有相同的元素个数（见make_softmax_layer()），其实由此也可知，
**                stride的值必然为1,不然output的元素个数肯定少于input的元素个数（所以对于softmax来说，感觉设置stride是没有必要的，有点自相矛盾的意思）
** 说明：上面的注释出现了新的量词单位，这里厘清一下关系：输入input中包括batch中所有图片的输入数据，其中一张图片具有inputs个元素，一张图片的元素又分成了groups组，
**      每组元素个数为n=l.inputs/l.groups
*/
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    // 遍历batch中的每张图片
    for(b = 0; b < batch; ++b){
        // 每张图片又按组遍历：一组一组遍历  这里的组应该是目标识别部分用到
        for(g = 0; g < groups; ++g){  // 每一个位置都需要写清楚
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

