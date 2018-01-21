#ifndef MATRIX_H
#define MATRIX_H
#include "darknet.h"
// 复制矩阵
matrix copy_matrix(matrix m);
// 打印输出矩阵当中每一个数值
void print_matrix(matrix m);

matrix hold_out_matrix(matrix *m, int n);
// 矩阵resize
matrix resize_matrix(matrix m, int size);

float *pop_column(matrix *m, int c);

#endif
