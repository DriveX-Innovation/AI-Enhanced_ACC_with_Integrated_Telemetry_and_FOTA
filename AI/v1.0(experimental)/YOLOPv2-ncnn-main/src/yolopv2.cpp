#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <chrono>
#include <string>

#define MAX_STRIDE 64

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// ─── ncnn helpers (unchanged from original) ──────────────────────────────────

static void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Crop");

    ncnn::ParamDict pd;
    ncnn::Mat axes = ncnn::Mat(1);  axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);  ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1); starts.fill(start);
    pd.set(9, starts);
    pd.set(10, ends);
    pd.set(11, axes);

    op->load_param(pd);
    op->create_pipeline(opt);
    op->forward(in, out, opt);
    op->destroy_pipeline(opt);
    delete op;
}

static void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");

    ncnn::ParamDict pd;
    pd.set(0, 2);        // resize_type (bilinear)
    pd.set(1, scale);    // height_scale
    pd.set(2, scale);    // width_scale
    pd.set(3, out_h);    // height
    pd.set(4, out_w);    // width

    op->load_param(pd);
    op->create_pipeline(opt);
    op->forward(in, out, opt);
    op->destroy_pipeline(opt);
    delete op;
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p) i++;
        while (faceobjects[j].prob < p) j--;

        if (i <= j)
        {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        { if (left < j) qsort_descent_inplace(faceobjects, left, j); }
        #pragma omp section
        { if (i < right) qsort_descent_inplace(faceobjects, i, right); }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty()) return;
    qsort_descent_inplace(faceobjects, 0, (int)faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = (int)faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = faceobjects[i].rect.area();

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold) keep = 0;
        }
        if (keep) picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride,
                                const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob,
                                float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;
    int num_grid_x, num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class   = feat_blob.w - 5;
    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score) { class_index = k; class_score = score; }
                }

                float confidence = sigmoid(featptr[4]) * sigmoid(class_score);
                if (confidence < prob_threshold) continue;

                float dx = sigmoid(featptr[0]);
                float dy = sigmoid(featptr[1]);
                float dw = sigmoid(featptr[2]);
                float dh = sigmoid(featptr[3]);

                float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                float pb_cy = (dy * 2.f - 0.5f + i) * stride;
                float pb_w  = powf(dw * 2.f, 2) * anchor_w;
                float pb_h  = powf(dh * 2.f, 2) * anchor_h;

                Object obj;
                obj.rect.x      = pb_cx - pb_w * 0.5f;
                obj.rect.y      = pb_cy - pb_h * 0.5f;
                obj.rect.width  = pb_w;
                obj.rect.height = pb_h;
                obj.label = class_index;
                obj.prob  = confidence;
                objects.push_back(obj);
            }
        }
    }
}

// ─── Draw results onto a frame (returns annotated image) ─────────────────────

static void draw_results(cv::Mat& image,
                         const std::vector<Object>& objects,
                         ncnn::Mat& da_seg_mask,
                         ncnn::Mat& ll_seg_mask,
                         double fps)
{
    // Segmentation overlay
    const float* da_ptr = (const float*)da_seg_mask.data;
    const float* ll_ptr = (const float*)ll_seg_mask.data;
    int sw = da_seg_mask.w;
    int sh = da_seg_mask.h;

    // Resize masks to match display frame if needed
    cv::Mat da_vis(sh, sw, CV_8UC1, cv::Scalar(0));
    cv::Mat ll_vis(sh, sw, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < sh; i++)
    {
        for (int j = 0; j < sw; j++)
        {
            if (da_ptr[i * sw + j] < da_ptr[sw * sh + i * sw + j])
                da_vis.at<uchar>(i, j) = 255;
            if (std::round(ll_ptr[i * sw + j]) == 1.f)
                ll_vis.at<uchar>(i, j) = 255;
        }
    }

    // Blend drivable area (green) and lane lines (blue) onto image
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            // map pixel to mask coordinates
            int mi = i * sh / image.rows;
            int mj = j * sw / image.cols;
            if (mi >= sh) mi = sh - 1;
            if (mj >= sw) mj = sw - 1;

            if (da_vis.at<uchar>(mi, mj))
                image.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 200, 0);   // green
            if (ll_vis.at<uchar>(mi, mj))
                image.at<cv::Vec3b>(i, j) = cv::Vec3b(200, 0, 0);   // blue
        }
    }

    // Bounding boxes
    for (const auto& obj : objects)
    {
        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 255), 2);

        char text[64];
        snprintf(text, sizeof(text), "%.1f%%", obj.prob * 100.f);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols) x = image.cols - label_size.width;

        cv::rectangle(image,
                      cv::Rect(x, y, label_size.width, label_size.height + baseLine),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // FPS counter
    char fps_text[32];
    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
    cv::putText(image, fps_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
}

// ─── Inference (model loaded once, reused per frame) ─────────────────────────

static ncnn::Net yolopv2_net;
static bool net_loaded = false;

static bool load_model(const char* model_dir)
{
    std::string param_path = std::string(model_dir) + "/yolopv2.param";
    std::string bin_path   = std::string(model_dir) + "/yolopv2.bin";

    if (yolopv2_net.load_param(param_path.c_str()) != 0)
    {
        fprintf(stderr, "Failed to load param: %s\n", param_path.c_str());
        return false;
    }
    if (yolopv2_net.load_model(bin_path.c_str()) != 0)
    {
        fprintf(stderr, "Failed to load model: %s\n", bin_path.c_str());
        return false;
    }
    net_loaded = true;
    fprintf(stdout, "Model loaded from %s\n", model_dir);
    return true;
}

static int detect_yolopv2(const cv::Mat& bgr,
                           std::vector<Object>& objects,
                           ncnn::Mat& da_seg_mask,
                           ncnn::Mat& ll_seg_mask)
{
    const int   target_size    = 640;
    const float prob_threshold = 0.30f;
    const float nms_threshold  = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox
    int w = img_w, h = img_h;
    float scale = 1.f;
    if (w > h) { scale = (float)target_size / w; w = target_size; h = (int)(h * scale); }
    else        { scale = (float)target_size / h; h = target_size; w = (int)(w * scale); }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolopv2_net.create_extractor();
    ex.input("images", in_pad);

    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("det0", out);
        ncnn::Mat anchors(6);
        anchors[0] = 12.f; anchors[1] = 16.f;
        anchors[2] = 19.f; anchors[3] = 36.f;
        anchors[4] = 40.f; anchors[5] = 28.f;
        std::vector<Object> objs8;
        generate_proposals(anchors, 8, in, out, prob_threshold, objs8);
        proposals.insert(proposals.end(), objs8.begin(), objs8.end());
    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("det1", out);
        ncnn::Mat anchors(6);
        anchors[0] = 36.f; anchors[1] = 75.f;
        anchors[2] = 76.f; anchors[3] = 55.f;
        anchors[4] = 72.f; anchors[5] = 146.f;
        std::vector<Object> objs16;
        generate_proposals(anchors, 16, in, out, prob_threshold, objs16);
        proposals.insert(proposals.end(), objs16.begin(), objs16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("det2", out);
        ncnn::Mat anchors(6);
        anchors[0] = 142.f; anchors[1] = 110.f;
        anchors[2] = 192.f; anchors[3] = 243.f;
        anchors[4] = 459.f; anchors[5] = 401.f;
        std::vector<Object> objs32;
        generate_proposals(anchors, 32, in, out, prob_threshold, objs32);
        proposals.insert(proposals.end(), objs32.begin(), objs32.end());
    }

    // segmentation masks
    {
        ncnn::Mat da, ll;
        ex.extract("677", da);
        ex.extract("769", ll);
        slice(da, da_seg_mask, hpad / 2, in_pad.h - hpad / 2, 1);
        slice(ll, ll_seg_mask, hpad / 2, in_pad.h - hpad / 2, 1);
        slice(da_seg_mask, da_seg_mask, wpad / 2, in_pad.w - wpad / 2, 2);
        slice(ll_seg_mask, ll_seg_mask, wpad / 2, in_pad.w - wpad / 2, 2);
        interp(da_seg_mask, 1.f / scale, 0, 0, da_seg_mask);
        interp(ll_seg_mask, 1.f / scale, 0, 0, ll_seg_mask);
    }

    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = (int)picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        float x0 = (objects[i].rect.x - wpad / 2) / scale;
        float y0 = (objects[i].rect.y - hpad / 2) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width  - wpad / 2) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - hpad / 2) / scale;

        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x      = x0;
        objects[i].rect.y      = y0;
        objects[i].rect.width  = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

// ─── main ────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  Image mode : %s image  <image_path>  [model_dir]\n"
        "  Camera mode: %s camera [camera_id]   [model_dir]\n"
        "\n"
        "  camera_id  : 0 for /dev/video0, 1 for /dev/video1, etc.  (default: 0)\n"
        "  model_dir  : directory containing yolopv2.param/.bin      (default: ../models)\n"
        "\n"
        "  Press 'q' or ESC to quit camera mode.\n"
        "  Press 's' to save a snapshot.\n",
        prog, prog);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return -1;
    }

    std::string mode      = argv[1];
    std::string model_dir = "../models";

    // ── IMAGE MODE ────────────────────────────────────────────────────────────
    if (mode == "image")
    {
        if (argc < 3) { print_usage(argv[0]); return -1; }
        if (argc >= 4) model_dir = argv[3];

        if (!load_model(model_dir.c_str())) return -1;

        cv::Mat frame = cv::imread(argv[2], 1);
        if (frame.empty())
        {
            fprintf(stderr, "Cannot read image: %s\n", argv[2]);
            return -1;
        }

        std::vector<Object> objects;
        ncnn::Mat da_seg_mask, ll_seg_mask;
        detect_yolopv2(frame, objects, da_seg_mask, ll_seg_mask);
        draw_results(frame, objects, da_seg_mask, ll_seg_mask, 0.0);

        cv::imwrite("result.jpg", frame);
        fprintf(stdout, "Saved result.jpg  (%d objects)\n", (int)objects.size());
        cv::imshow("YOLOPv2 result", frame);
        cv::waitKey(0);
        return 0;
    }

    // ── CAMERA MODE ───────────────────────────────────────────────────────────
    if (mode == "camera")
    {
        int cam_id = 0;
        if (argc >= 3) cam_id = atoi(argv[2]);
        if (argc >= 4) model_dir = argv[3];

        if (!load_model(model_dir.c_str())) return -1;

        cv::VideoCapture cap;

        // Try V4L2 backend first (native on Pi), fall back to default
        cap.open(cam_id, cv::CAP_V4L2);
if (!cap.isOpened())
{
    fprintf(stderr, "Retrying with default backend...\n");
    cap.open(cam_id);
}
if (!cap.isOpened())
{
    fprintf(stderr, "Cannot open camera %d\n", cam_id);
    return -1;
}

// Force MJPG to prevent USB bandwidth saturation
cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
cap.set(cv::CAP_PROP_FPS, 30);


        int actual_w   = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int actual_h   = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double actual_fps = cap.get(cv::CAP_PROP_FPS);
        fprintf(stdout, "Camera opened: %dx%d @ %.0f fps\n", actual_w, actual_h, actual_fps);
        fprintf(stdout, "Press 'q' / ESC to quit, 's' to save snapshot.\n");

        int snapshot_count = 0;
        double fps = 0.0;

        while (true)
        {
            auto t0 = std::chrono::steady_clock::now();

            cv::Mat frame;
            if (!cap.read(frame) || frame.empty())
            {
                fprintf(stderr, "Camera read error – retrying...\n");
                cv::waitKey(30);
                continue;
            }

            std::vector<Object> objects;
            ncnn::Mat da_seg_mask, ll_seg_mask;
            detect_yolopv2(frame, objects, da_seg_mask, ll_seg_mask);
            draw_results(frame, objects, da_seg_mask, ll_seg_mask, fps);

            cv::imshow("YOLOPv2 – Pi5 Camera  (q=quit  s=snapshot)", frame);

            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            fps = 1.0 / (elapsed > 1e-6 ? elapsed : 1e-6);

            int key = cv::waitKey(1);
            if (key == 'q' || key == 27)   // q or ESC
                break;
            if (key == 's')
            {
                char fname[64];
                snprintf(fname, sizeof(fname), "snapshot_%03d.jpg", snapshot_count++);
                cv::imwrite(fname, frame);
                fprintf(stdout, "Snapshot saved: %s\n", fname);
            }
        }

        cap.release();
        cv::destroyAllWindows();
        return 0;
    }

    // ── UNKNOWN MODE ──────────────────────────────────────────────────────────
    fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
    print_usage(argv[0]);
    return -1;
}
