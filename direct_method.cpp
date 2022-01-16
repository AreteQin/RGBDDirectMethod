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
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;
using namespace Eigen;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 2, 1> Vec2;

// G2O
/// vertex used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    virtual void setToOriginImpl() override
    {
        _estimate = Sophus::SE3d();
    }
    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override
    {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }
    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out) const override {}
};
/// 定义仅优化位姿的一元边，2表示观测值的维度，Vec2表示观测值的数据类型是一个2×1的向量，VertexPose表示定点的数据类型
class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Vec2, VertexPose>
{
public:
    EdgeProjectionPoseOnly(const cv::Mat &img1_,
                           const cv::Mat &img2_,
                           const VecVector2d &px_ref_,
                           const vector<double> depth_ref_,
                           const cv::Mat &depth_img2_, ) : px_ref(px_ref_), depth_ref(depth_ref_), depth_img2(depth_img2_), img1(img1_), img2(img2_)
    virtual void computeError() override {
        const VertexPose *V static_cast<VertexPose *> (_vertices[0]); // _vertices[0]表示这条边所链接的地一个顶点，由于是一元边，因此只有_vertices[1]，若是二元边则还会存在_vertices[1]
        
    }
}

// Camera intrinsics
//double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
double fx = 726.28741455078,
       fy = 726.28741455078, cx = 354.6496887207, cy = 186.46566772461;

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
        Sophus::SE3d &T21_) : px_ref(px_ref_), depth_ref(depth_ref_), depth_img2(depth_img2_), T21(T21_), img1(img1_), img2(img2_)
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
    // const cv::Mat &img1,
    // const cv::Mat &img2,
    cv::Mat img1,
    cv::Mat img2,
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

void showPointCloud(
    const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

void showPointCloud(
    const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

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
    boost::format fmt("./%s/%06d.png"); //图像文件格式

    TrajectoryType ground_truth_poses;

    ifstream fin("./pose.txt");
    if (!fin)
    {
        cerr << "请在有pose.txt的目录下运行此程序" << endl;
        return 1;
    }
    vector<cv::Mat> color_images, depth_images;
    for (int i = 0; i < 5; i++)
    {
        boost::format fmt("./%s/%d.%s");
        color_images.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str(), cv::IMREAD_COLOR));
        depth_images.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), cv::IMREAD_ANYDEPTH));

        double data[7] = {0};
        for (auto &d : data)
            fin >> d;
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                          Eigen::Vector3d(data[0], data[1], data[2]));
        ground_truth_poses.push_back(pose);
    }

    // generate pixels in ref and load depth data
    VecVector2d pixels_ref;
    vector<double> depth_ref;
    for (int k = 0; k < color_images[0].rows; k++)
    {
        for (int h = 0; h < color_images[0].cols; h++)
        {
            double depth1 = depth_images[0].at<uchar>(k, h);
            if (depth1 < min_depth || depth1 > max_depth)
                continue;
            depth_ref.push_back(depth1);
            pixels_ref.push_back(Eigen::Vector2d(k, h));
        }
    }

    // for (int i = 0; i < nPoints; i++)
    // {
    //     int x = rng.uniform(boarder, color_images[0].cols - boarder); // don't pick pixels close to boarder
    //     int y = rng.uniform(boarder, color_images[0].rows - boarder); // don't pick pixels close to boarder
    //     double depth = depth_images[0].at<uchar>(y, x);
    //     depth_ref.push_back(depth);
    //     pixels_ref.push_back(Eigen::Vector2d(x, y));
    // }

    std::cout << "points generated" << endl;

    // estimates 01~05.png's pose using this information
    //vector<Sophus::SE3d> poses;
    TrajectoryType poses;
    Sophus::SE3d T_cur_ref;
    Sophus::SE3d T_0;
    poses.push_back(T_0);

    // for (int i = 0; i < depth_images[1].rows; i++)
    // {
    //     for (int j = 0; j < depth_images[1].cols; j++)
    //     {
    //         double depth11 = depth_images[1].at<uchar>(i, j);
    //         cout<< depth11<<" ";
    //     }
    // }
    // cv::imshow("img1", color_images[0]);
    // cv::waitKey(0);
    // cv::imshow("img2", color_images[1]);
    // cv::waitKey(0);
    // cv::imshow("depth1", depth_images[0]);
    // cv::waitKey(0);
    // cv::imshow("depth2", depth_images[1]);
    // cv::waitKey(0);
    // cout<<"channel of color_images[1]: "<<color_images[1].rows<<endl;
    // cout<<"channel of depth_images[1]: "<<depth_images[1].rows<<endl;
    // cout<<"format of color_images[1]: "<<color_images[1].cols<<endl;
    // cout<<"format of depth_images[1]: "<<depth_images[1].cols<<endl;
    // for (int k = 0; k < depth_images[1].rows; k++)
    // {
    //     for (int h = 0; h < depth_images[1].cols; h++)
    //     {
    //         uint8_t *data = (uint8_t *)depth_images[1].data;
    //         std::cout << unsigned(data[h + depth_images[1].step * k]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // try single layer by uncomment this line
    // DirectPoseEstimationSingleLayer(color_images[1], color_images[2], pixels_ref, depth_ref, T_cur_ref);
    DirectPoseEstimationMultiLayer(color_images[0], color_images[1], pixels_ref, depth_ref, depth_images[1], T_cur_ref);

    //poses.push_back(T_cur_ref);
    poses.push_back(T_cur_ref);
    double depthScale = 1000.0;
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000);

    for (int i = 0; i < 2; i++)
    {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = color_images[i];
        cv::Mat depth = depth_images[i];
        Sophus::SE3d T = poses[i];
        cout << "T" << i << ": " << endl
             << T.matrix() << endl;
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++)
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0)
                    continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;
                //cout<<"point "<<u+v<<": "<<endl<<pointWorld<< endl;

                Vector6d p;
                p.head<3>() = pointWorld;
                p[5] = color.data[v * color.step + u * color.channels()];     // blue
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                pointcloud.push_back(p);
                //cout<<p<<endl<<endl;
            }
    }

    cout << "点云共有" << pointcloud.size() << "个点." << endl;

    showPointCloud(pointcloud);
}

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    const cv::Mat &depth_img2,
    Sophus::SE3d &T21)
{
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true);     // 打开调试输出

    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(T21);
    optimizer.addVertex(vertex_pose);

    // edges counter
    int index = 1;

    //新增部分：第一个相机作为顶点连接的边
    for (size_t i = 0; i < px_ref.size(); ++i)
    {
        EdgeProjection *edge = new EdgeProjection(px_ref[i], depth_ref[i], img1, img2);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(GetPixelValue(img1, px_ref[i][0], px_ref[i][1]));
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, depth_img2, T21);
    // cout << "Jacobian Accumulator generated" << endl;

    for (int iter = 0; iter < iterations; iter++)
    {
        jaco_accu.reset();
        //cout << "Jacobian Accumulator reset" << endl;
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
                if ((GetPixelValue(depth_img2, u + x, v + y) <= min_depth) || (GetPixelValue(depth_img2, u + x, v + y) >= max_depth))
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
    // const cv::Mat &img1,
    // const cv::Mat &img2,
    cv::Mat img1,
    cv::Mat img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    const cv::Mat &depth_img2,
    Sophus::SE3d &T21)
{
    cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);
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

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud)
{

    if (pointcloud.empty())
    {
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

    while (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p : pointcloud)
        {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000); // sleep 5 ms
    }
    return;
}

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud)
{

    if (pointcloud.empty())
    {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST); //启用深度缓存。
    glEnable(GL_BLEND);      //启用gl_blend混合。Blend混合是将源色和目标色以某种方式混合生成特效的技术。
    //混合常用来绘制透明或半透明的物体。在混合中起关键作用的α值实际上是将源色和目标色按给定比率进行混合，以达到不同程度的透明。
    //α值为0则完全透明，α值为1则完全不透明。混合操作只能在RGBA模式下进行，颜色索引模式下无法指定α值。
    //物体的绘制顺序会影响到OpenGL的混合处理。
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); //混合函数。参数1是源混合因子，参数2时目标混合因子。本命令选择了最常使用的参数。

    //定义投影和初始模型视图矩阵
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        //对应为gluLookAt,摄像机位置,参考点位置,up vector(上向量)
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));
    //管理OpenGl视口的位置和大小
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                //使用混合分数/像素坐标（OpenGl视图坐标）设置视图的边界
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                //指定用于接受键盘或鼠标输入的处理程序
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false)
    {
        //清除屏幕
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //激活要渲染到视图
        d_cam.Activate(s_cam);
        //glClearColor：red、green、blue、alpha分别是红、绿、蓝、不透明度，值域均为[0,1]。
        //即设置颜色，为后面的glClear做准备，默认值为（0,0,0,0）。切记：此函数仅仅设定颜色，并不执行清除工作。
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        //glPointSize 函数指定栅格化点的直径。一定要在要在glBegin前,或者在画东西之前。
        glPointSize(2);
        //glBegin()要和glEnd()组合使用。其参数表示创建图元的类型，GL_POINTS表示把每个顶点作为一个点进行处理
        glBegin(GL_POINTS);
        for (auto &p : pointcloud)
        {
            glColor3f(p[3], p[3], p[3]);  //在OpenGl中设置颜色
            glVertex3d(p[0], p[1], p[2]); //设置顶点坐标
        }
        glEnd();
        pangolin::FinishFrame(); //结束
        usleep(5000);            // sleep 5 ms
    }
    return;
}
//     glEnable(GL_DEPTH_TEST);
//     glEnable(GL_BLEND);
//     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

//     pangolin::OpenGlRenderState s_cam(
//         pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
//         pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

//     pangolin::View &d_cam = pangolin::CreateDisplay()
//                                 .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
//                                 .SetHandler(new pangolin::Handler3D(s_cam));

//     while (pangolin::ShouldQuit() == false)
//     {
//         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//         d_cam.Activate(s_cam);
//         glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

//         glPointSize(2);
//         glBegin(GL_POINTS);
//         for (auto &p : pointcloud)
//         {
//             glColor3f(p[3], p[3], p[3]);
//             glVertex3d(p[0], p[1], p[2]);
//         }
//         glEnd();
//         pangolin::FinishFrame();
//         usleep(5000); // sleep 5 ms
//     }
//     return;
// }
