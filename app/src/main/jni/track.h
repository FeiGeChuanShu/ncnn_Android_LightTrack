// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef TRACK_H
#define TRACK_H

#include <opencv2/core/core.hpp>

#include <net.h>

class LightTrack
{
public:
    int load(const char* modeltype, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, bool use_gpu = false);

    int detect(const cv::Mat& rgb);
    void update(ncnn::Net& net, cv::Mat x_crop,ncnn::Mat zf, cv::Point2f &target_pos,
                 cv::Point2f &target_sz, float scale_z, std::vector<float> &cls_score);
    ncnn::Mat get_template(ncnn::Net& net, cv::Mat templateImage,cv::Point2f pos);
    int draw(cv::Mat& rgb);
    int init(int x, int y, int width, int height, int screenWidth);
    int isInit(){return is_init;}
private:
    ncnn::Net light_track;
    ncnn::Mat zf;
    cv::Rect template_box;
    cv::Mat temp;
    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
    std::vector<float> window;
    cv::Point2f target_sz;
    cv::Point2f target_pos;
    int is_init = 0;
    const float mean_vals1[3] = { 123.675f, 116.28f,  103.53f };
    const float norm_vals1[3] = { 0.01712475f, 0.0175f, 0.01742919f };
};

#endif // TRACK_H
