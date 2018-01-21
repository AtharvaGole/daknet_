#ifndef BOX_H
#define BOX_H
#include "darknet.h"
/**
 * 用于目标识别的4个参数 中心点坐标(x, y) 宽和长坐标
 * 注意4个值都是比例坐标, 即占整个图片宽高的比例
 * */
typedef struct{
    float dx, dy, dw, dh;
} dbox;

float box_rmse(box a, box b);

dbox diou(box a, box b);

box decode_box(box b, box anchor);

box encode_box(box b, box anchor);

#endif
