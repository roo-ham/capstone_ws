#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std::chrono_literals;

class BallTrackerNode : public rclcpp::Node {
public:
    BallTrackerNode() : Node("ball_tracker_node"), running_(true) {
        cap_.open(0, cv::CAP_V4L2);
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap_.set(cv::CAP_PROP_FPS, 30); 

        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "카메라를 열 수 없습니다.");
            throw std::runtime_error("Camera init failed");
        }

        marker_floor_pts_ = {
            cv::Point2f(-0.13f,  0.115f),
            cv::Point2f( 0.13f,  0.115f),
            cv::Point2f(-0.055f, -0.115f),
            cv::Point2f( 0.055f, -0.115f)
        };

        dt_ = 1.0 / 300.0;
        X_ = Eigen::Vector4d::Zero();
        P_ = Eigen::Matrix4d::Identity();
        
        F_ = Eigen::Matrix4d::Identity();
        F_(0, 2) = dt_;
        F_(1, 3) = dt_;

        H_ = Eigen::Matrix4d::Identity();

        Q_ = Eigen::Matrix4d::Zero();
        Q_(0,0) = 0.0001; Q_(1,1) = 0.0001;
        Q_(2,2) = 0.01;   Q_(3,3) = 0.01;

        R_ = Eigen::Matrix4d::Zero();
        R_(0,0) = 0.0001; R_(1,1) = 0.0001;
        R_(2,2) = 0.01;   R_(3,3) = 0.01;

        // 4. 공유 메모리 (5차원으로 확장: x, y, vx, vy, timestamp)
        int fd_shm = shm_open("ball_state_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(fd_shm, 5 * sizeof(double));
        shm_ptr_ = (double*)mmap(0, 5 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm, 0);
        std::fill(shm_ptr_, shm_ptr_ + 5, 0.0);

        kf_timer_ = this->create_wall_timer(
            std::chrono::microseconds(3333), 
            std::bind(&BallTrackerNode::process_kalman_filter, this)
        );

        cam_thread_ = std::thread(&BallTrackerNode::camera_loop, this);

        RCLCPP_INFO(this->get_logger(), "C++ Ball Tracker Node (with Timestamp SHM) Started.");
    }

    ~BallTrackerNode() {
        running_ = false;
        if (cam_thread_.joinable()) cam_thread_.join();
        // 메모리 해제 크기 5로 수정
        munmap(shm_ptr_, 5 * sizeof(double));
        shm_unlink("ball_state_shm");
    }

private:
    std::atomic<bool> running_;
    std::thread cam_thread_;
    std::mutex kf_lock_;
    rclcpp::TimerBase::SharedPtr kf_timer_;
    cv::VideoCapture cap_;

    std::vector<cv::Point2f> marker_floor_pts_;
    cv::Mat homography_matrix_;

    Eigen::Vector4d X_;
    Eigen::Matrix4d P_, F_, H_, Q_, R_;
    double dt_;

    double* shm_ptr_ = nullptr;
    
    cv::Point2f last_ball_pos_ = cv::Point2f(0.0f, 0.0f);
    double last_cam_time_ = 0.0;
    bool has_last_pos_ = false;

    #define TAU 0.3  

    std::vector<cv::Point2f> input_yellow_;       
    std::vector<cv::Point2f> filtered_yellow_;    
    bool yellow_initialized_ = false;             
    double last_tick_time_ = 0.0;                 

    void camera_loop() {
        cv::Mat frame, hsv;
        std::vector<cv::Mat> hsv_channels(3);

        while (running_ && rclcpp::ok()) {
            if (!cap_.read(frame) || frame.empty()) continue;

            double current_time = this->get_clock()->now().nanoseconds() / 1e9;
            if (last_tick_time_ == 0.0) last_tick_time_ = current_time;
            double dt = current_time - last_tick_time_;
            last_tick_time_ = current_time;

            cv::Mat rgb;
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

            cv::Mat red_mask = cv::Mat::zeros(rgb.size(), CV_8UC1);
            cv::Mat yellow_mask = cv::Mat::zeros(rgb.size(), CV_8UC1);

            for (int y = 0; y < rgb.rows; ++y) {
                cv::Vec3b* rgb_ptr = rgb.ptr<cv::Vec3b>(y);
                uchar* r_mask_ptr = red_mask.ptr<uchar>(y);
                uchar* y_mask_ptr = yellow_mask.ptr<uchar>(y);

                for (int x = 0; x < rgb.cols; ++x) {
                    int r_dist = std::abs((int) rgb_ptr[x][0] - 230) + std::abs((int) rgb_ptr[x][1] - 60) + std::abs((int) rgb_ptr[x][2] - 60);
                    if (r_dist <= 110) r_mask_ptr[x] = 255;

                    int y_dist = std::abs((int) rgb_ptr[x][0] - 180) + std::abs((int) rgb_ptr[x][1] - 210) + std::abs((int) rgb_ptr[x][2] - 80);
                    if (y_dist <= 110) y_mask_ptr[x] = 255;
                }
            }

            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::erode(red_mask, red_mask, kernel);
            cv::erode(yellow_mask, yellow_mask, kernel);

            auto red_cents = get_centroids(red_mask, 1);
            auto yellow_cents = get_centroids(yellow_mask, 4);

            if (yellow_cents.size() == 4) {
                std::sort(yellow_cents.begin(), yellow_cents.end(), [](cv::Point2f a, cv::Point2f b) { return a.y < b.y; });
                std::vector<cv::Point2f> top_two = {yellow_cents[0], yellow_cents[1]};
                std::vector<cv::Point2f> bottom_two = {yellow_cents[2], yellow_cents[3]};
                
                std::sort(top_two.begin(), top_two.end(), [](cv::Point2f a, cv::Point2f b) { return a.x < b.x; });
                std::sort(bottom_two.begin(), bottom_two.end(), [](cv::Point2f a, cv::Point2f b) { return a.x < b.x; });

                input_yellow_ = {top_two[0], top_two[1], bottom_two[0], bottom_two[1]};
                
                if (!yellow_initialized_) {
                    filtered_yellow_ = input_yellow_;
                    yellow_initialized_ = true;
                }
            }

            if (yellow_initialized_) {
                double k = (dt + TAU > 0) ? (dt / (dt + TAU)) : 1.0;
                if (dt <= 0) k = 0.0; 

                for (size_t i = 0; i < 4; ++i) {
                    filtered_yellow_[i].x = k * input_yellow_[i].x + (1.0 - k) * filtered_yellow_[i].x;
                    filtered_yellow_[i].y = k * input_yellow_[i].y + (1.0 - k) * filtered_yellow_[i].y;
                }

                homography_matrix_ = cv::findHomography(filtered_yellow_, marker_floor_pts_);
            }

            cv::Point2f ball_floor_pos;
            bool ball_found = false;

            if (!homography_matrix_.empty() && red_cents.size() == 1) {
                std::vector<cv::Point2f> ball_cam_pts = {red_cents[0]};
                std::vector<cv::Point2f> ball_floor_pts;
                cv::perspectiveTransform(ball_cam_pts, ball_floor_pts, homography_matrix_);
                ball_floor_pos = ball_floor_pts[0];

                double vx_meas = 0.0, vy_meas = 0.0;

                if (has_last_pos_) {
                    double dt_cam = current_time - last_cam_time_;
                    if (dt_cam > 0) {
                        vx_meas = (ball_floor_pos.x - last_ball_pos_.x) / dt_cam;
                        vy_meas = (ball_floor_pos.y - last_ball_pos_.y) / dt_cam;
                    }
                }

                last_ball_pos_ = ball_floor_pos;
                last_cam_time_ = current_time;
                has_last_pos_ = true;

                Eigen::Vector4d Z(ball_floor_pos.x, ball_floor_pos.y, vx_meas, vy_meas);
                
                // [수정] 5번째 원소에 관측된 시간 기록
                shm_ptr_[0] = Z(0); shm_ptr_[1] = Z(1); shm_ptr_[2] = Z(2); shm_ptr_[3] = Z(3);
                shm_ptr_[4] = current_time; 

                std::lock_guard<std::mutex> lock(kf_lock_);
                Eigen::Vector4d y_res = Z - H_ * X_;
                Eigen::Matrix4d S = H_ * P_ * H_.transpose() + R_;
                Eigen::Matrix4d K = P_ * H_.transpose() * S.inverse();
                X_ = X_ + K * y_res;
                
                P_ = (Eigen::Matrix4d::Identity() - K * H_) * P_; 
                ball_found = true;
            } else {
                has_last_pos_ = false;
            }
        }
    }

    void process_kalman_filter() {
        std::lock_guard<std::mutex> lock(kf_lock_);
        X_ = F_ * X_;
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    std::vector<cv::Point2f> get_centroids(const cv::Mat& mask, size_t max_count) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) > cv::contourArea(b);
        });

        std::vector<cv::Point2f> centroids;
        for (size_t i = 0; i < std::min(contours.size(), max_count); ++i) {
            cv::Moments M = cv::moments(contours[i]);
            if (M.m00 > 0) {
                centroids.push_back(cv::Point2f(M.m10 / M.m00, M.m01 / M.m00));
            }
        }
        return centroids;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BallTrackerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}