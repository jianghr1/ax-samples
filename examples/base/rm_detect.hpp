/*
 * AXERA is pleased to support the open source community by making ax-samples available.
 * 
 * Copyright (c) 2022, AXERA Semiconductor (Shanghai) Co., Ltd. All rights reserved.
 * 
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 * 
 * https://opensource.org/licenses/BSD-3-Clause
 * 
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

/*
 * Author: ls.wang
 */

#pragma once

#include <cstdint>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
namespace rm_detection
{

    typedef enum Color {
        Red=0,
        Blue=1,
        Gray=2,
    } Color;

    typedef struct Object
    {
        cv::Rect_<float> rect;
        Color color;
        int label;
        float prob;
        cv::Point2f landmark[4];
    } Object;

    inline static float clamp(
        float val,
        float min = 0.f,
        float max = 1536.f)
    {
        return val > min ? (val < max ? val : max) : min;
    }

    inline float fast_exp(const float& x)
    {
        union
        {
            uint32_t i;
            float f;
        } v{};
        v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
        return v.f;
    }

    inline float fast_sigmoid(const float& x)
    {
        return 1.0f / (1.0f + fast_exp(-x));
    }

    
    static inline float sigmoid(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

    inline static float fast_softmax(
        const float* src,
        float* dst,
        int length)
    {
        const float alpha = *std::max_element(src, src + length);
        float denominator = 0;
        float dis_sum = 0;
        for (int i = 0; i < length; ++i)
        {
            dst[i] = fast_exp(src[i] - alpha);
            denominator += dst[i];
        }
        for (int i = 0; i < length; ++i)
        {
            dst[i] /= denominator;
            dis_sum += i * dst[i];
        }
        return dis_sum;
    }

    template<typename T>
    static inline float intersection_area(const T& a, const T& b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    template<typename T>
    static void qsort_descent_inplace(std::vector<T>& faceobjects, int left, int right)
    {
        int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].prob;

        while (i <= j)
        {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j)
            {
                // swap
                std::swap(faceobjects[i], faceobjects[j]);

                i++;
                j--;
            }
        }
#pragma omp parallel sections
        {
#pragma omp section
            {
                if (left < j) qsort_descent_inplace(faceobjects, left, j);
            }
#pragma omp section
            {
                if (i < right) qsort_descent_inplace(faceobjects, i, right);
            }
        }
    }

    template<typename T>
    static void qsort_descent_inplace(std::vector<T>& faceobjects)
    {
        if (faceobjects.empty())
            return;

        qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
    }

    template<typename T>
    static void nms_sorted_bboxes(const std::vector<T>& faceobjects, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();

        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].rect.area();
        }

        for (int i = 0; i < n; i++)
        {
            const T& a = faceobjects[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const T& b = faceobjects[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }

    static void generate_proposals_yolov8_rm_native(int stride, const float* box, const float* cls, const float* kps, float prob_threshold, std::vector<Object>& objects,
                                                      int letterbox_cols, int letterbox_rows, const int num_point = 4, int cls_num = 11)
    {
        int feat_w = letterbox_cols / stride;
        int feat_h = letterbox_rows / stride;
        for (int h = 0; h <= feat_h - 1; h++)
        {
            for (int w = 0; w <= feat_w - 1; w++)
            {
                // process cls score
                float color_score = -100;
                int color_index = 0;
                for (int s = 0; s < 3; s++)
                {
                    float score = cls[s];
                    if (score > color_score)
                    {
                        color_index = s;
                        color_score = score;
                    }
                }

                float id_score = -100;
                int id_index = 0;
                for (int s = 3; s < cls_num; s++)
                {
                    float score = cls[s];
                    if (score > id_score)
                    {
                        id_index = s - 3;
                        id_score = score;
                    }
                }

                float box_prob = sigmoid(id_score/100) * sigmoid(color_score/100);
                if (box_prob > prob_threshold)
                {
                    float pb_cx = (w + 0.5f) * stride;
                    float pb_cy = (h + 0.5f) * stride;

                    float x0 = pb_cx - box[0];
                    float y0 = pb_cy - box[1];
                    float x1 = pb_cx + box[2];
                    float y1 = pb_cy + box[3];

                    x0 = std::max(std::min(x0, (float)(letterbox_cols - 1)), 0.f);
                    y0 = std::max(std::min(y0, (float)(letterbox_rows - 1)), 0.f);
                    x1 = std::max(std::min(x1, (float)(letterbox_cols - 1)), 0.f);
                    y1 = std::max(std::min(y1, (float)(letterbox_rows - 1)), 0.f);

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.color = (Color)color_index;
                    obj.label = id_index;
                    obj.prob = box_prob;
                    for (int k = 0; k < 4; k++)
                    {
                        obj.landmark[k].x = (kps[k * 2 + 0] * 2.f + w) * stride;
                        obj.landmark[k].y = (kps[k * 2 + 1] * 2.f + h) * stride;
                    }
                    objects.push_back(obj);
                }
                box += 4;
                cls += cls_num;
                kps += 2 * num_point;
            }
        }
    }

    static void draw_keypoints(const cv::Mat& bgr, const std::vector<Object>& objects,
                               const char* output_name)
    {
        cv::Mat image = bgr.clone();

        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];

            fprintf(stdout, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], color: %d, id: %d\n", obj.label, obj.prob * 100, obj.rect.x,
                    obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, (int)obj.color, obj.label);

            cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

            std::string text = std::to_string(obj.label) + std::string(": ") + std::to_string(int(obj.prob*100));
            
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;
            
            cv::Scalar white(255, 255, 255);
            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), white, -1);

            cv::Scalar obj_color;
            switch (obj.color)
            {
                case rm_detection::Color::Blue:
                    obj_color = cv::Scalar(255, 0, 0);
                    break;
                case rm_detection::Color::Red:
                    obj_color = cv::Scalar(0, 0, 255);
                    break;
                case rm_detection::Color::Gray:
                    obj_color = cv::Scalar(50, 50, 50);
                    break;
            }
            cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, obj_color);
            cv::line(image, obj.landmark[0], obj.landmark[1], white, 2);
            cv::line(image, obj.landmark[1], obj.landmark[2], white, 2);
            cv::line(image, obj.landmark[2], obj.landmark[3], white, 2);
            cv::line(image, obj.landmark[3], obj.landmark[0], white, 2);

        }
        cv::imwrite(std::string(output_name) + ".jpg", image);
    }

    void get_out_bbox_kps(std::vector<Object>& proposals, std::vector<Object>& objects, const float nms_threshold, int letterbox_rows, int letterbox_cols, int src_rows, int src_cols)
    {
        qsort_descent_inplace(proposals);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        /* yolov8 draw the result */
        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / src_rows) < (letterbox_cols * 1.0 / src_cols))
        {
            scale_letterbox = letterbox_rows * 1.0 / src_rows;
        }
        else
        {
            scale_letterbox = letterbox_cols * 1.0 / src_cols;
        }
        resize_cols = int(scale_letterbox * src_cols);
        resize_rows = int(scale_letterbox * src_rows);

        int tmp_h = (letterbox_rows - resize_rows) / 2;
        int tmp_w = (letterbox_cols - resize_cols) / 2;

        float ratio_x = (float)src_rows / resize_rows;
        float ratio_y = (float)src_cols / resize_cols;

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];
            float x0 = (objects[i].rect.x);
            float y0 = (objects[i].rect.y);
            float x1 = (objects[i].rect.x + objects[i].rect.width);
            float y1 = (objects[i].rect.y + objects[i].rect.height);

            x0 = (x0 - tmp_w) * ratio_x;
            y0 = (y0 - tmp_h) * ratio_y;
            x1 = (x1 - tmp_w) * ratio_x;
            y1 = (y1 - tmp_h) * ratio_y;

            x0 = std::max(std::min(x0, (float)(src_cols - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(src_rows - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(src_cols - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(src_rows - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;

            for (int j = 0; j < 4; j++)
            {
                objects[i].landmark[j].x -= tmp_w;
                objects[i].landmark[j].x *= ratio_x;
                objects[i].landmark[j].y -= tmp_h;
                objects[i].landmark[j].y *= ratio_y;
            }
        }
    }
    
} // namespace detection
