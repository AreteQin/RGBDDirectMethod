#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
//double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
double fx = 726.28741455078, fy = 726.28741455078, cx = 354.6496887207, cy = 186.46566772461;

double min_depth = 1.0;
double max_depth = 255.0;

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/// class for accumulator jacobians in parallel
class JacobianAccumulator
{
public:
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,
        const vector<double> depth_ref_,
        const cv::Mat &depth_img2_,
        Sophus::SE3d &T21_) : img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), depth_img2(depth_img2_), T21(T21_)
    {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }

    /// accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    /// get hessian matrix
    Matrix6d hessian() const { return H; }

    /// get bias
    Vector6d bias() const { return b; }

    /// get total cost
    double cost_func() const { return cost; }

    /// get projected points
    VecVector2d projected_points() const { return projection; }

    /// reset h, b, cost to zero
    void reset()
    {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const cv::Mat &depth_img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    const cv::Mat &depth_img2,
    Sophus::SE3d &T21);

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    const cv::Mat &depth_img2,
    Sophus::SE3d &T21);

// 得到3D点后，在像素坐标中定位后，获取该像素的值
float GetPixelValue(const cv::Mat &img, float x, float y)
{
    // boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols)
        x = img.cols - 1;
    if (y >= img.rows)
        y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)]; // 定位到做对比的像素位置
    float xx = x - floor(x);
    float yy = y - floor(y);
    // 使用bilinear interpolation计算该位置的近似灰度
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]);
}

int main()
{
    cv::Mat color_img1 = cv::imread("./color_img1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat color_img2 = cv::imread("./color_img2.png", cv::IMREAD_GRAYSCALE);
    cv::Mat depth_img1 = cv::imread("./depth_img1.png", cv::IMREAD_ANYDEPTH);
    cv::Mat depth_img2 = cv::imread("./depth_img2.png", cv::IMREAD_ANYDEPTH);

    // generate pixels in ref and load depth data
    VecVector2d pixels_ref;
    vector<double> depth_ref;
    for (int k = 0; k < color_img1.rows; k++)
    {
        for (int h = 0; h < color_img1.cols; h++)
        {
            double depth1 = depth_img1.at<uchar>(k, h);
            if (depth1 < min_depth || depth1 > max_depth)
                continue;
            depth_ref.push_back(depth1);
            pixels_ref.push_back(Eigen::Vector2d(k, h));
        }
    }

    // for (int i = 0; i < nPoints; i++)
    // {
    //     int x = rng.uniform(boarder, color_img1.cols - boarder); // don't pick pixels close to boarder
    //     int y = rng.uniform(boarder, color_img1.rows - boarder); // don't pick pixels close to boarder
    //     double depth = depth_img1.at<uchar>(y, x);
    //     depth_ref.push_back(depth);
    //     pixels_ref.push_back(Eigen::Vector2d(x, y));
    // }

    std::cout << "points generated" << endl;

    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    // cv::Mat depth_mat = disparity_mat;
    // for (int i = 0; i < disparity_mat.rows; i++)
    // {
    //     for (int j = 0; j < disparity_mat.cols; j++)
    //     {
    //         int disparity = disparity_mat.at<uchar>(i, j);
    //         double depth = fx * baseline / disparity; // you know this is disparity to depth
    //         uchar value = (uchar)depth;
    //         depth_mat.ptr<uchar>(i)[j] = value;
    //     }
    // }
    // cv::imshow("img", color_img1);
    // cv::waitKey(0);
    // cv::imshow("img", color_img2);
    // cv::waitKey(0);
    // cv::imshow("img", depth_img1);
    // cv::waitKey(0);
    // cv::imshow("img", depth_mat);
    // cv::waitKey(0);
    // cout<<"channel of color_img1: "<<color_img1.rows<<endl;
    // cout<<"channel of depth_img1: "<<depth_img1.rows<<endl;
    // cout<<"format of color_img1: "<<color_img1.cols<<endl;
    // cout<<"format of depth_img1: "<<depth_img1.cols<<endl;
    // for (int k = 0; k < depth_img1.rows; k++)
    // {
    //     for (int h = 0; h < depth_img1.cols; h++)
    //     {
    //         uint8_t *data = (uint8_t *)depth_img1.data;
    //         std::cout << unsigned(data[h + depth_img1.step * k]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // try single layer by uncomment this line
    // DirectPoseEstimationSingleLayer(color_img1, color_img2, pixels_ref, depth_ref, T_cur_ref);
    DirectPoseEstimationMultiLayer(color_img1, color_img2, pixels_ref, depth_ref, depth_img2, T_cur_ref);
}

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    const cv::Mat &depth_img2,
    Sophus::SE3d &T21)
{

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, depth_img2, T21);
    // cout << "Jacobian Accumulator generated" << endl;

    for (int iter = 0; iter < iterations; iter++)
    {
        jaco_accu.reset();
        // cout << "Jacobian Accumulator reset" << endl;
        cv::parallel_for_(cv::Range(0, px_ref.size()), std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1)); // 多线程计算Jacobian,Hessian and b

        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0]))
        {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost)
        {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3)
        {
            // converge
            break;
        }

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n"
         << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i)
    {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0)
        {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    cv::namedWindow("current", cv::WINDOW_NORMAL); // 自适应窗口大小
    cv::imshow("current", img2_show);
    cv::waitKey();
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range)
{

    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++)
    {
        // cout << "points loop ============" << endl;

        // compute the projection in the second image
        Eigen::Vector3d point_ref = depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0) // depth invalid
            continue;
        float u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy; // pixel position in the second image calculated

        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size || v > img2.rows - half_patch_size)
            continue; // skip the points out of sight

        projection[i] = Eigen::Vector2d(u, v);

        // cout << "pixel position in the second image calculated" << endl;

        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
               Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++)
            {
                double error;
                if ( (GetPixelValue(depth_img2, u + x, v + y) <= min_depth ) || ( GetPixelValue(depth_img2, u + x, v + y) >= max_depth) )
                {
                    error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) - GetPixelValue(img2, u + x, v + y);
                }
                else
                {
                    error = (GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) - GetPixelValue(img2, u + x, v + y)) + (depth_ref[i] - GetPixelValue(depth_img2, u + x, v + y));
                }

                // double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) - GetPixelValue(img2, u + x, v + y);

                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y)));

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }
    // cout << "points loop end here =============================================" << endl;

    if (cnt_good)
    {
        // set hessian, bias and cost
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    const cv::Mat &depth_img2,
    Sophus::SE3d &T21)
{

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++)
    {
        if (i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            cv::Mat img1_pyr, img2_pyr;
            // resize images according to pyramid level
            cv::resize(pyr1[i - 1], img1_pyr, cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr, cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    // cout << "pyramid images Generated" << endl;

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy; // backup the old values
    for (int level = pyramids - 1; level >= 0; level--)
    {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px : px_ref)
        {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        // cout << "into signal layer Direct method" << endl;
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, depth_img2, T21);
    }
}
