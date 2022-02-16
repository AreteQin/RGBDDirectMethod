#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>
#include <fstream>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
// #include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;
using namespace Eigen;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 2, 1> Vec2;

double min_depth = 1.0;
double max_depth = 255.0;
const int half_patch_size = 1;
// Camera intrinsics
double fx = 726.28741455078, fy = 726.28741455078, cx = 354.6496887207, cy = 186.46566772461;

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// cast char to string
void tokenize(std::string const &str, const char delim,
              std::vector<std::string> &out) {
    size_t start;
    size_t end = 0;

    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

// 得到3D点在图像中像素坐标后，获取该像素的灰度或深度
double GetPixelValue(const cv::Mat &img, double x, double y) {
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
    double xx = x - floor(x);
    double yy = y - floor(y);
    // 使用bilinear interpolation计算该位置的近似灰度
    return double(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]);
}

double TukeysBiweight(double error, double threshold_c) {
    if (abs(error) > threshold_c) {
        return 0;
    } else {
        return abs(error * (1 - error * error / (threshold_c * threshold_c)) *
                   (1 - error * error / (threshold_c * threshold_c)));
    }
}

// G2O
/// vertex used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    /// left multiplication on SE3
    void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    bool read(istream &in) override {}

    bool write(ostream &out) const override {}
};

/// 定义仅优化位姿的一元边，2表示观测值的维度，Vec2表示观测值的数据类型是一个2×1的向量，VertexPose表示定点的数据类型
class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Vec2, VertexPose> {
public:
    EdgeProjectionPoseOnly(const cv::Mat &img1_,
                           const cv::Mat &img2_,
                           const Eigen::Vector2d &px_ref_,
                           const double &depth_ref_,
                           const cv::Mat &depth_img2_) : img1(img1_), img2(img2_), px_ref(px_ref_),
                                                         depth_ref(depth_ref_), depth_img2(depth_img2_) {}

    void computeError() override {
        const VertexPose *v = dynamic_cast<VertexPose *>(_vertices[0]); // _vertices[0]表示这条边所链接的地一个顶点，由于是一元边，因此只有_vertices[0]，若是二元边则还会存在_vertices[1]
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d position_in_ref_cam = depth_ref *
                                              Eigen::Vector3d((px_ref[0] - cx) / fx, (px_ref[1] - cy) / fy,
                                                              1); // 深度乘以归一化坐标就得到了相机坐标系下的三维点
        Eigen::Vector3d position_in_cur_cam = T * position_in_ref_cam;
        // cout<<"position_in_cur_cam: "<<endl<<position_in_cur_cam<<endl;
        // cout<< fx <<", "<<position_in_cur_cam[1] <<", "<<position_in_cur_cam[2]<<", "<< cx<<endl;
        double u_in_cur_pixel = fx * position_in_cur_cam[1] / position_in_cur_cam[2] + cx;
        double v_in_cur_pixel = fy * position_in_cur_cam[0] / position_in_cur_cam[2] + cy;
        double I2 = GetPixelValue(img2, u_in_cur_pixel, v_in_cur_pixel);
        double Z2 = static_cast<unsigned short>(depth_img2.data[int(u_in_cur_pixel) * depth_img2.step +
                                                                int(v_in_cur_pixel)]);
        if (v_in_cur_pixel < 1 || v_in_cur_pixel > depth_img2.cols - 1 || u_in_cur_pixel < 1 ||
            u_in_cur_pixel > depth_img2.rows - 1) // out of sight in current frame
        {
            _error(0, 0) = 0.0;
            _error(1, 0) = 0.0;
        } else if (Z2 < 1 || Z2 > 255) {
            _error(0, 0) = I2 - _measurement(0, 0);
            _error(1, 0) = 0;
        } else {
            _error(0, 0) = I2 - _measurement(0, 0);
            _error(1, 0) = Z2 - _measurement(1, 0);
        }
        // _error(0, 0) = I2 - _measurement(0, 0);
        // _error(1, 0) = 10 * (depth_in_cur_cam_ - position_in_cur_cam[2]);
        // std::cout << _error(0, 0) << ", " << _error(1, 0) << ". "<<std::endl;
        _error(1, 0) = TukeysBiweight(_error(1, 0), 20);
        _error(0, 0) = TukeysBiweight(_error(0, 0), 50);
        //     std::cout<<"current_depth_.col, row: "<<depth_img2.cols<<", "<<depth_img2.rows<<std::endl;
        //     std::cout<<"u,v: "<<u_in_cur_pixel<<", "<<v_in_cur_pixel<<std::endl;
        //     std::cout<<"I1: "<<_measurement(0,0)<<std::endl;
        //     std::cout<<"I2: "<<I2<<std::endl;
        //     std::cout<<"Zwrap: "<<_measurement(1,0)<<std::endl;
        //     std::cout<<"Z2: "<<Z2<<std::endl;
        //     std::cout <<"errors_tukeys: "<< _error(0, 0) << ", " << _error(1, 0) << ". "<<std::endl;

    }

    void linearizeOplus() override // 重写线性化函数，即得到泰勒展开e(x+delta_x)=e(x)+J^T*delta_x中的J，推导过程见14讲7.3.3
    {
        const VertexPose *v = dynamic_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d position_in_ref_cam = depth_ref *
                                              Eigen::Vector3d((px_ref[0] - cx) / fx, (px_ref[1] - cy) / fy,
                                                              1); // 深度乘以归一化坐标就得到了相机坐标系下的三维点
        Eigen::Vector3d position_in_cur_cam = T * position_in_ref_cam;
        double u_in_cur_pixel = fx * position_in_cur_cam[0] / position_in_cur_cam[2] + cx;
        double v_in_cur_pixel = fy * position_in_cur_cam[1] / position_in_cur_cam[2] + cy;

        Eigen::Matrix<double, 1, 6> J_1, J_2, J_position_xi_Z;
        double X = position_in_cur_cam[0], Y = position_in_cur_cam[1], Z = position_in_cur_cam[2];
        double Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        Matrix26d J_position_xi;
        Eigen::Vector2d J_color_gradient, J_depth_gradient;

        J_position_xi(0, 0) = fx * Z_inv;
        J_position_xi(0, 1) = 0;
        J_position_xi(0, 2) = -fx * X * Z2_inv;
        J_position_xi(0, 3) = -fx * X * Y * Z2_inv;
        J_position_xi(0, 4) = fx + fx * X * X * Z2_inv;
        J_position_xi(0, 5) = -fx * Y * Z_inv;

        J_position_xi(1, 0) = 0;
        J_position_xi(1, 1) = fy * Z_inv;
        J_position_xi(1, 2) = -fy * Y * Z2_inv;
        J_position_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
        J_position_xi(1, 4) = fy * X * Y * Z2_inv;
        J_position_xi(1, 5) = fy * X * Z_inv;

        J_position_xi_Z(0, 0) = 0;
        J_position_xi_Z(0, 1) = 0;
        J_position_xi_Z(0, 2) = 1;
        J_position_xi_Z(0, 3) = -Y;
        J_position_xi_Z(0, 4) = X;
        J_position_xi_Z(0, 5) = 0;

        J_color_gradient = Eigen::Vector2d(
                0.5 * (GetPixelValue(img2, u_in_cur_pixel + 1, v_in_cur_pixel) -
                       GetPixelValue(img2, u_in_cur_pixel - 1, v_in_cur_pixel)),
                0.5 * (GetPixelValue(img2, u_in_cur_pixel, v_in_cur_pixel + 1) -
                       GetPixelValue(img2, u_in_cur_pixel, v_in_cur_pixel - 1)));

        J_depth_gradient = Eigen::Vector2d(
                0.5 * (GetPixelValue(depth_img2, u_in_cur_pixel + 1, v_in_cur_pixel) -
                       GetPixelValue(depth_img2, u_in_cur_pixel - 1, v_in_cur_pixel)),
                0.5 * (GetPixelValue(depth_img2, u_in_cur_pixel, v_in_cur_pixel + 1) -
                       GetPixelValue(depth_img2, u_in_cur_pixel, v_in_cur_pixel - 1)));

        J_1 = (J_color_gradient.transpose() * J_position_xi);
        J_2 = (J_depth_gradient.transpose() * J_position_xi) - J_position_xi_Z;

        // total jacobian
        _jacobianOplusXi << J_1[0], J_1[1], J_1[2], J_1[3], J_1[4], J_1[5],
                J_2[0], J_2[1], J_2[2], J_2[3], J_2[4], J_2[5];
    }

    bool read(istream &in) override {}

    bool write(ostream &out) const override {}

private:
    const Eigen::Vector2d px_ref;
    const double depth_ref;
    const cv::Mat img1, img2, depth_img2;
};

void DirectPoseEstimationMultiLayer(
        // const cv::Mat &img1,
        // const cv::Mat &img2,
        cv::Mat img1,
        cv::Mat img2,
        const VecVector2d &px_ref,
        const vector<double> &depth_ref,
        const cv::Mat &depth_img2,
        Sophus::SE3d &T21);

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> &depth_ref,
        const cv::Mat &depth_img2,
        Sophus::SE3d &T21);

void showPointCloud(
        const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

void showPointCloud(
        const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main() {

    string dataset_path_ = "/home/qin/Downloads/mannequin_face_2";
    std::ifstream dataset_path(dataset_path_ + "/calibration.txt");

    if (!dataset_path) {
        cout << "cannot find " << dataset_path_ << "/calibration.txt";
    }

    // read images
    std::ifstream image_stamps(dataset_path_ + "/associated.txt");
    vector<cv::Mat> depth_images_, color_images_;
    if (!image_stamps) {
        std::cout << "cannot find " << dataset_path_ << "/associated.txt" << std::endl;
    } else // file exists
    {
        std::string line;
        while (getline(image_stamps, line)) {
            std::vector<std::string> each_in_line;
            tokenize(line, ' ', each_in_line);
            depth_images_.push_back(cv::imread((dataset_path_ + "/" + each_in_line[3]), cv::IMREAD_UNCHANGED));
            color_images_.push_back(cv::imread((dataset_path_ + "/" + each_in_line[1]), cv::IMREAD_COLOR));
        }
    }
    image_stamps.close();
    std::cout << depth_images_.size() << " pairs of images found." << std::endl;

    vector<VecVector2d> pixels_refs;
    vector<vector<double>> depth_refs;
    // generate pixels in ref and load depth data
    for (int i = 0; i < depth_images_.size()-1; i++) {
        VecVector2d pixels_ref;
        vector<double> depth_ref;
        for (int k = 0; k < color_images_[i].rows; k++) {
            for (int h = 0; h < color_images_[i].cols; h++) {
                double depth1 = depth_images_[i].at<uchar>(k, h);
                if (depth1 < min_depth || depth1 > max_depth)
                    continue;
                depth_ref.push_back(depth1);
                pixels_ref.push_back(Eigen::Vector2d(k, h));
            }
        }
        pixels_refs.push_back(pixels_ref);
        depth_refs.push_back(depth_ref);
        std::cout << "image " << i << " processed" << endl;
    }

    std::cout << "points generated" << endl;

    TrajectoryType poses;
    Sophus::SE3d T_cur_ref;
    Sophus::SE3d T_0;
    poses.push_back(T_0);

    DirectPoseEstimationSingleLayer(color_images_[0], color_images_[1], pixels_refs[0],
                                    depth_refs[0],
                                    depth_images_[1],
                                    T_cur_ref);
    poses.push_back(T_cur_ref);

    for (int i = 2; i < depth_images_.size()-1; i++) {
        DirectPoseEstimationSingleLayer(color_images_[i - 1], color_images_[i], pixels_refs[i - 1],
                                        depth_refs[i - 1],
                                        depth_images_[i],
                                        T_cur_ref);
//        DirectPoseEstimationMultiLayer(color_images[i-1], color_images[i], pixels_refs[i-1], depth_refs[i-1],
//                                       depth_images[i],
//                                       T_cur_ref);
        //poses.push_back(T_cur_ref);
        poses.push_back(T_cur_ref * poses[i - 1]);
        cout << "T"<< i-1 <<"= \n"
                << (T_cur_ref * poses[i - 1]).matrix() << endl;
    }
    cout << "poses calculated" << endl;

    double depthScale = 1000.0;
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000);

    for (int i = 0; i < depth_images_.size()-1; i++) {
        cout << "generating point cloud: " << i + 1 << endl;
        cv::Mat color = color_images_[i];
        cv::Mat depth = depth_images_[i];
        Sophus::SE3d T = poses[i];
        cout << "T" << i << ": " << endl
             << T.matrix() << endl;
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0)
                    continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;
                Vector6d p;
                p.head<3>() = pointWorld;
                p[5] = color.data[v * color.step + u * color.channels()];     // blue
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                pointcloud.push_back(p);
            }
    }

    cout << "generated" << pointcloud.size() << "points" << endl;

    showPointCloud(pointcloud);
}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> &depth_ref,
        const cv::Mat &depth_img2,
        Sophus::SE3d &T21) {
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(false);    // 打开调试输出

    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(T21);
    //vertex_pose->setMarginalized(true);
    optimizer.addVertex(vertex_pose);

    // edges counter
    int index = 1;

    //新增部分：第一个相机作为顶点连接的边
    for (size_t i = 0; i < px_ref.size(); ++i) {
        EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(img1, img2, px_ref[i], depth_ref[i],
                                                                  depth_img2);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);

        Eigen::Vector3d position_in_ref_cam = depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx,
                                                                             (px_ref[i][1] - cy) / fy,
                                                                             1); // 深度乘以归一化坐标就得到了相机坐标系下的三维点
        Eigen::Vector3d position_in_cur_cam = vertex_pose->estimate() * position_in_ref_cam;
        // double u_in_cur_pixel = fx * position_in_cur_cam[0] / position_in_cur_cam[2] + cx;
        // double v_in_cur_pixel = fy * position_in_cur_cam[1] / position_in_cur_cam[2] + cy;

        Eigen::Matrix<double, 2, 1> measurements;
        measurements << GetPixelValue(img1, px_ref[i][0], px_ref[i][1]), position_in_cur_cam[2];
        //cout<<"measurements: " <<measurements<<endl;
        edge->setMeasurement(measurements);
        edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);
    T21 = vertex_pose->estimate();

    // // plot the projected pixels here
    // cv::Mat img2_show;
    // //cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    // //VecVector2d projection = jaco_accu.projected_points();
    // //VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    // vector<Eigen::Vector2d> projection,new_px_ref;
    // for (int i = 0; i < depth_ref.size(); ++i)
    // {
    //         Eigen::Vector3d point_ref = depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
    //         Eigen::Vector3d point_cur = T21 * point_ref;
    //         if (point_cur[2] < 0) // depth invalid
    //             continue;
    //         double u = fx * point_cur[0] / point_cur[2] + cx;
    //         double v = fy * point_cur[1] / point_cur[2] + cy; // pixel position in the second image calculated
    //         if (u < half_patch_size || u > depth_img2.cols - half_patch_size || v < half_patch_size || v > depth_img2.rows - half_patch_size || GetPixelValue(depth_img2, u, v) <= min_depth || GetPixelValue(depth_img2, u, v) >= max_depth)
    //             continue; // skip the points out of sight
    //         new_px_ref.push_back(px_ref[i]);
    //         projection.push_back( Eigen::Vector2d(u, v) );
    // }
    // for (size_t i = 0; i < new_px_ref.size(); ++i)
    // {
    //     auto p_ref = px_ref[i];
    //     auto p_cur = projection[i];
    //     if (p_cur[0] > 0 && p_cur[1] > 0)
    //     {
    //         cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
    //         cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
    //                  cv::Scalar(0, 250, 0));
    //     }
    // }
    // cv::namedWindow("current", cv::WINDOW_NORMAL); // 自适应窗口大小
    // cv::imshow("current", img2_show);
    // cv::waitKey();
}

void DirectPoseEstimationMultiLayer(
        // const cv::Mat &img1,
        // const cv::Mat &img2,
        cv::Mat img1,
        cv::Mat img2,
        const VecVector2d &px_ref,
        const vector<double> &depth_ref,
        const cv::Mat &depth_img2,
        Sophus::SE3d &T21) {
    cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);
    // parameters
    int pyramids = 5;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2, depth_pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
            depth_pyr2.push_back(depth_img2);
        } else {
            cv::Mat img1_pyr, img2_pyr, depth_pyramid;
            // resize images according to pyramid level
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            cv::resize(depth_pyr2[i - 1], depth_pyramid, cv::Size(depth_pyr2[i - 1].cols * pyramid_scale,
                                                                  depth_pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
            depth_pyr2.push_back(depth_pyramid);
        }
    }
    // cout << "pyramid images Generated" << endl;

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy; // backup the old values
    for (int level = pyramids - 2; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        cout << "fxG: " << fxG << ", scale: " << scales[level] << ", level: " << level << endl;
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        // cout << "into signal layer Direct method" << endl;
        cout << "fx,fy,cx,cy: " << fx << ", " << fy << ", " << cx << ", " << cy << ", " << endl;
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, depth_pyr2[level],
                                        T21);
    }
}

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000); // sleep 5 ms
    }
    return;
}