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

#include "track.h"
#include <android/log.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <android/log.h>
#include "cpu.h"

int LightTrack::load(AAssetManager* mgr, const char* modeltype, bool use_gpu)
{
    light_track.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    light_track.opt = ncnn::Option();

#if NCNN_VULKAN
    light_track.opt.use_vulkan_compute = use_gpu;
#endif

    light_track.opt.num_threads = ncnn::get_big_cpu_count();

   // char parampath[256];
   // char modelpath[256];
   /// sprintf(parampath, "scrfd_%s-opt2.param", modeltype);
    //sprintf(modelpath, "scrfd_%s-opt2.bin", modeltype);

    light_track.load_param(mgr, "lighttrack-op.param");
    light_track.load_model(mgr, "lighttrack-op.bin");

    grid_to_search_x.resize(16 * 16, 0);
    grid_to_search_y.resize(16 * 16, 0);
    window.resize(16*16,0);
    return 0;
}


static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static std::vector<float> sc_f(std::vector<float> w, std::vector<float> h,float sz)
{
    std::vector<float> pad(16 * 16, 0);
    std::vector<float> sz2;
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            pad[i*16+j] = (w[i * 16 + j] + h[i * 16 + j]) * 0.5;
        }
    }
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            float t = std::sqrt((w[i * 16 + j] + pad[i*16+j]) * (h[i * 16 + j] + pad[i*16+j])) / sz;

            sz2.push_back(std::max(t,(float)1.0/t) );
        }
    }


    return sz2;
}

static std::vector<float> rc_f(std::vector<float> w, std::vector<float> h, cv::Point2f target_sz)
{
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2;
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            float t = ratio / (w[i * 16 + j] / h[i * 16 + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }

    return sz2;
}
static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}
cv::Mat get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz)
{
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = std::round(pos.x - c);
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = std::round(pos.y - c);
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;
    cv::Mat im_path_original;

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);
        //te_im(cv::Rect(left_pad, top_pad, im.cols, im.rows)) = im;
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, 0.f);
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));

    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));

    return im_path;
}

void LightTrack::update(ncnn::Net& net, cv::Mat x_crop,ncnn::Mat zf_, cv::Point2f &target_pos_,
             cv::Point2f &target_sz_, float scale_z, std::vector<float> &cls_score)
{
    target_sz_.x = target_sz_.x * scale_z;
    target_sz_.y = target_sz_.y * scale_z;

    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat ncnn_in1 = ncnn::Mat::from_pixels(x_crop.data, ncnn::Mat::PIXEL_RGB, x_crop.cols, x_crop.rows);
    ncnn_in1.substract_mean_normalize(mean_vals1, norm_vals1);


    ex.input("input", ncnn_in1);
    ex.input("temp", zf_);

    ncnn::Mat cls, reg;
    ex.extract("cls", cls);
    ex.extract("reg", reg);

    float* cls_data = (float*)cls.data;
    cls_score.clear();
    for (int i = 0; i < 16 * 16; i++)
    {
        cls_score.push_back(sigmoid(cls_data[i]));
    }

    std::vector<float> pred_x1(16*16,0), pred_y1(16 * 16, 0), pred_x2(16 * 16, 0), pred_y2(16 * 16, 0);

    float* reg_data1 = reg.channel(0);
    float* reg_data2 = reg.channel(1);
    float* reg_data3 = reg.channel(2);
    float* reg_data4 = reg.channel(3);
    for (int j = 0; j < 16; j++)
    {
        for (int k = 0; k < 16; k++)
        {
            pred_x1[j * 16 + k] = grid_to_search_x[j * 16 + k] - reg_data1[j * 16 + k];
            pred_y1[j * 16 + k] = grid_to_search_y[j * 16 + k] - reg_data2[j * 16 + k];
            pred_x2[j * 16 + k] = grid_to_search_x[j * 16 + k] + reg_data3[j * 16 + k];
            pred_y2[j * 16 + k] = grid_to_search_y[j * 16 + k] + reg_data4[j * 16 + k];
        }
    }

    std::vector<float> w(16 * 16, 0), h(16 * 16, 0);
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            w[i * 16 + j] = pred_x2[i * 16 + j] - pred_x1[i * 16 + j];
            h[i * 16 + j] = pred_y2[i * 16 + j] - pred_y1[i * 16 + j];
        }
    }

    float sz_wh = sz_whFun(target_sz_);
    std::vector<float> s_c = sc_f(w, h, sz_wh);
    std::vector<float> r_c = rc_f(w, h, target_sz_);

    std::vector<float> penalty(16*16,0);
    for (int i = 0; i < 16 * 16; i++)
    {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i]-1) * 0.062);
    }

    std::vector<float> pscore(16*16,0);
    int r_max = 0, c_max = 0;
    float maxScore = 0;
    for (int i = 0; i < 16 * 16; i++)
    {
        pscore[i] = (penalty[i] * cls_score[i]) * (1 - 0.225) + window[i] * 0.225;
        if (pscore[i] > maxScore)
        {
            maxScore = pscore[i];
            r_max = std::floor(i / 16);
            c_max = ((float)i / 16 - r_max) * 16;
        }
    }

    float predx1 = pred_x1[r_max * 16 + c_max];
    float predy1 = pred_y1[r_max * 16 + c_max];
    float predx2 = pred_x2[r_max * 16 + c_max];
    float predy2 = pred_y2[r_max * 16 + c_max];

    float pred_xs = (predx1 + predx2) / 2;
    float pred_ys = (predy1 + predy2) / 2;
    float pred_w = predx2 - predx1;
    float pred_h = predy2 - predy1;

    float diff_xs = pred_xs - 256 / 2;
    float diff_ys = pred_ys - 256 / 2;

    diff_xs = diff_xs / scale_z;
    diff_ys = diff_ys / scale_z;
    pred_w = pred_w / scale_z;
    pred_h = pred_h / scale_z;

    target_sz_.x = target_sz_.x / scale_z;
    target_sz_.y = target_sz_.y / scale_z;

    float lr = penalty[r_max * 16 + c_max] * cls_score[r_max * 16 + c_max] * 0.765;
    float res_xs = target_pos_.x + diff_xs;
    float res_ys = target_pos_.y + diff_ys;
    float res_w = pred_w * lr + (1 - lr) * target_sz_.x;
    float res_h = pred_h * lr + (1 - lr) * target_sz_.y;

    target_pos_.x = res_xs;
    target_pos_.y = res_ys;
    target_sz_.x = target_sz_.x * (1 - lr) + lr * res_w;
    target_sz_.y = target_sz_.y * (1 - lr) + lr * res_h;

}
ncnn::Mat LightTrack::get_template(ncnn::Net& net, cv::Mat templateImage,cv::Point2f pos)
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            grid_to_search_x[i*16+j] = j*16;
            grid_to_search_y[i * 16 + j] = i * 16;
        }
    }

    std::vector<float> hanning(16,0);
    for (int i = 0; i < 16; i++)
    {
        float w = 0.5 - 0.5 * std::cos(2 * 3.1415926535898 *i / 15);
        hanning[i] = w;
    }
    for (int i = 0; i < 16; i++)
    {

        for (int j = 0; j < 16; j++)
        {
            window[i*16+j] = hanning[i] * hanning[j];
        }
    }
    float hc_z = target_sz.y + 0.5 * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + 0.5 * (target_sz.x + target_sz.y);
    float s_z = std::sqrt(wc_z * hc_z);
    cv::Mat temp = get_subwindow_tracking(templateImage, pos, 127, s_z);

    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat ncnn_in1 = ncnn::Mat::from_pixels(temp.data, ncnn::Mat::PIXEL_RGB, 127, 127);
    ncnn_in1.substract_mean_normalize(mean_vals1, norm_vals1);

    ex.input("input", ncnn_in1);
    ncnn::Mat zf_;
    ex.extract("745", zf_);

    return zf_;
}

int LightTrack::detect(const cv::Mat& rgb)
{
    float hc_z = target_sz.y + 0.5 * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + 0.5 * (target_sz.x + target_sz.y);
    float s_z = std::sqrt(wc_z * hc_z);
    float scale_z = 127. / s_z;
    float d_search = (256. - 127.) / 2;
    float pad = d_search / scale_z;
    int s_x = std::round(s_z + 2 * pad);

    cv::Mat x_crop = get_subwindow_tracking(rgb, target_pos, 256, s_x);

    std::vector<float> cls_score;
    update(light_track, x_crop, zf, target_pos, target_sz, scale_z,cls_score);

    target_pos.x = std::max(0.f, std::min((float)rgb.cols, target_pos.x));
    target_pos.y = std::max(0.f, std::min((float)rgb.rows, target_pos.y));
    target_sz.x = std::max(10.f, std::min((float)rgb.cols, target_sz.x));
    target_sz.y = std::max(10.f, std::min((float)rgb.rows, target_sz.y));

    return 0;
}

int LightTrack::draw(cv::Mat& rgb)
{
    if(!is_init)
    {
        rgb.copyTo(temp);
    }

    cv::rectangle(rgb, cv::Rect(target_pos.x - target_sz.x / 2, target_pos.y - target_sz.y / 2, target_sz.x, target_sz.y),
                  cv::Scalar(255, 0, 0), 2, 8);
    return 0;
}
int LightTrack::init(int x, int y, int width, int height,int screenWidth)
{
    if(!is_init)
    {

        float scale =(double) temp.cols/(double) screenWidth;
        template_box.x = x*scale;
        template_box.y = y*scale;
        template_box.width = width*scale;
        template_box.height = height*scale;
        target_pos = cv::Point2f(template_box.x + template_box.width / 2, template_box.y + template_box.height / 2);
        target_sz.x = template_box.width;
        target_sz.y = template_box.height;

        zf = get_template(light_track,temp,target_pos);
        is_init = 1;
    }

    return 0;
}
