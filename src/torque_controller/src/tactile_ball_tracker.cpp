#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <thread>
#include <mutex>
#include <cmath>
#include <chrono>
#include <std_msgs/msg/empty.hpp> // 추가됨: Empty 메시지 헤더

using json = nlohmann::json;
using namespace std::chrono_literals;

struct SensorData {
    double force = 0.0;
    double last_area = 0.0;
    double zero_area = 0.0;
    
    double k_val = 100.0; // k * 0.01 (double로 변경)
    double b_val = 100.0; // b - 100 (Zero Force를 위해 double로 변경)
    int target_fps = 60; // 프레임 제한
    
    std::vector<cv::Point> trap_pts; // 관심 구역 4개 꼭짓점
    cv::Mat mask;                    // 1회용으로 만들어둘 정적 마스크

    int thresh = 120;
    double current_fps = 0.0;
};

class TactileBallTracker : public rclcpp::Node {
public:
    TactileBallTracker() : Node("tactile_ball_tracker") {
        last_time_ = this->now(); 
        
        load_config();
        init_shm();
        
        // ROS 2 파라미터 선언 (원격 또는 터미널에서 실시간 조절 가능)
        this->declare_parameter("fps_limit", 60);
        this->declare_parameter("filter_alpha", 1.0); // 1.0 = 지연 없는 즉각 반응, 0.35 = 기존 느린 반응

        // --- 추가된 부분: Set Force Zero 구독 ---
        zero_sub_ = this->create_subscription<std_msgs::msg::Empty>(
            "set_force_zero", 10,
            [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
                std::lock_guard<std::mutex> lock(data_mutex_);
                for (int i = 0; i < 3; ++i) {
                    sensors_[i].b_val = -sensors_[i].last_area;
                }
                RCLCPP_INFO(this->get_logger(), "Set Force Zero Triggered. 'b' parameter updated.");
            });
        // ----------------------------------------

        // 3개의 카메라 스레드 시작
        for (int i = 0; i < 3; ++i) {
            running_ = true;
            capture_threads_.emplace_back(&TactileBallTracker::camera_thread_func, this, i);
        }

        // 상태 계산 및 SHM 전송 타이머 (100Hz)
        timer_ = this->create_wall_timer(
            10ms, std::bind(&TactileBallTracker::calculate_and_publish, this));

        // 터미널 UI 출력 타이머 (10Hz)
        ui_timer_ = this->create_wall_timer(
            100ms, std::bind(&TactileBallTracker::update_terminal_ui, this));
            
        RCLCPP_INFO(this->get_logger(), "Optimized Tactile Ball Tracker Node Started.");
    }

    ~TactileBallTracker() {
        running_ = false;
        for (auto& th : capture_threads_) {
            if (th.joinable()) th.join();
        }
        
        munmap(eef_ptr_, 6 * sizeof(double));
        close(fd_eef_);
        munmap(shm_ptr_, 5 * sizeof(double));
        close(fd_shm_);
        shm_unlink("ball_state_shm");
    }

private:
    double* shm_ptr_; 
    double* eef_ptr_; 
    int fd_shm_, fd_eef_;

    std::vector<std::thread> capture_threads_;
    bool running_;
    SensorData sensors_[3];
    std::string camera_sources_[3];
    
    std::mutex data_mutex_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr ui_timer_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr zero_sub_; // 구독자 선언

    double last_ball_x_ = 0.0;
    double last_ball_y_ = 0.0;
    rclcpp::Time last_time_;

    void init_shm() {
        fd_eef_ = shm_open("eef_pos_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(fd_eef_, 6 * sizeof(double));
        eef_ptr_ = (double*)mmap(0, 6 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_eef_, 0);

        shm_unlink("ball_state_shm");
        fd_shm_ = shm_open("ball_state_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(fd_shm_, 5 * sizeof(double));
        shm_ptr_ = (double*)mmap(0, 5 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm_, 0);
        std::fill(shm_ptr_, shm_ptr_ + 5, 0.0);
    }

    void load_config() {
        std::ifstream file("tactile_config.json");
        if (file.is_open()) {
            json j;
            file >> j;
            for (int i = 0; i < 3; ++i) {
                camera_sources_[i] = j["cameras"][i]["source"].get<std::string>();
                sensors_[i].k_val = j["cameras"][i]["k_slider"];
                sensors_[i].b_val = j["cameras"][i]["b_slider"];
                sensors_[i].zero_area = j["cameras"][i]["zero_area"];
                
                sensors_[i].trap_pts.clear();
                for(auto& pt : j["cameras"][i]["trap_src"]) {
                    sensors_[i].trap_pts.push_back(cv::Point(pt[0], pt[1]));
                }
            }
            file.close();
        }
    }

    void update_terminal_ui() {
        // ROS 파라미터에서 값 동적 읽기
        int global_fps_limit = this->get_parameter("fps_limit").as_int();
        for(int i=0; i<3; ++i) sensors_[i].target_fps = global_fps_limit;

        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // ANSI Escape 코드를 사용하여 터미널 화면 지우기 및 커서 홈 이동 (TUI 구성)
        printf("\033[H\033[J");
        printf("====================================================\n");
        printf("         OPTIMIZED TACTILE BALL TRACKER TUI         \n");
        printf("====================================================\n");
        printf(" Target FPS Limit: %d (Adjust via ROS Param)\n\n", global_fps_limit);
        
        for(int i=0; i<3; ++i) {
            printf(" [Sensor %d] FPS: %5.1f | Score(Area): %7.1f | Force: %6.2f\n", 
                   i+1, sensors_[i].current_fps, sensors_[i].last_area, sensors_[i].force);
        }
        
        printf("\n");
        printf(" [Estimated Ball Position]\n");
        printf(" X Pos: %7.2f\n", last_ball_x_);
        printf(" Y Pos: %7.2f\n", last_ball_y_);
        printf("====================================================\n");
        
        // 터미널 버퍼 강제 비우기
        fflush(stdout);
    }

    void camera_thread_func(int idx) {
        std::string source = camera_sources_[idx];
        
        auto process_frame = [&](cv::Mat& frame) {
            cv::Mat binary;

            // 1. 프레임 크기를 알게 된 '최초 1회'에만 정적 마스크(Static Mask) 생성
            if (sensors_[idx].mask.empty() && !sensors_[idx].trap_pts.empty()) {
                sensors_[idx].mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                // 관심 구역 내부만 255(흰색)로 채움
                cv::fillConvexPoly(sensors_[idx].mask, sensors_[idx].trap_pts, cv::Scalar(255));
            }

            // 2. 원본 프레임 전체를 지정한 thresh 값으로 즉시 이진화
            cv::threshold(frame, binary, sensors_[idx].thresh, 255, cv::THRESH_BINARY);

            // 3. 비트 연산(AND)으로 관심 구역 바깥의 픽셀을 순식간에 검은색(0)으로 삭제
            if (!sensors_[idx].mask.empty()) {
                cv::bitwise_and(binary, sensors_[idx].mask, binary);
            }

            // 4. 남은 흰색 픽셀 개수 세기 (초고속 연산)
            double total_area = cv::countNonZero(binary);

            // 5. 필터 및 힘 계산
            double alpha = this->get_parameter("filter_alpha").as_double();
            double smoothed_area = alpha * total_area + (1.0 - alpha) * sensors_[idx].last_area;
            
            std::lock_guard<std::mutex> lock(data_mutex_);
            double k = sensors_[idx].k_val;
            double b = sensors_[idx].b_val;

            sensors_[idx].last_area = smoothed_area;
            sensors_[idx].force = (k * (smoothed_area+b));
            
            // if(sensors_[idx].force < 0) sensors_[idx].force = 0.0;
        };

        if (source == "0" || source == "1") {
            // timeout 설정 제거 및 해상도 최적화. std::endl 버퍼 방지 
            std::string cmd = "rpicam-vid -t 0 --camera " + source + 
                              " --width 640 --height 400 --framerate 120" + 
                              " --codec yuv420 --denoise off --nopreview -o - 2>/dev/null";
            
            FILE* pipe = popen(cmd.c_str(), "r");
            if (!pipe) return;

            int width = 640, height = 400;
            int y_size = width * height;           
            int frame_size = y_size * 3 / 2;       
            
            std::vector<uint8_t> buffer(frame_size);
            cv::Mat frame(height, width, CV_8UC1); 

            auto last_frame_time = std::chrono::steady_clock::now();
            int frame_count = 0;

            while (running_ && rclcpp::ok()) {
                auto loop_start = std::chrono::steady_clock::now();

                size_t bytes_read = fread(buffer.data(), 1, frame_size, pipe);
                if (bytes_read != (size_t)frame_size) continue;

                std::memcpy(frame.data, buffer.data(), y_size);
                process_frame(frame);

                // 실시간 FPS 계산 로직
                frame_count++;
                auto now = std::chrono::steady_clock::now();
                double elapsed_sec = std::chrono::duration<double>(now - last_frame_time).count();
                if (elapsed_sec >= 1.0) {
                    sensors_[idx].current_fps = frame_count / elapsed_sec;
                    frame_count = 0;
                    last_frame_time = now;
                }

                // 동적 프레임 제한 (CPU 점유율 절약)
                double target_ms = 1000.0 / sensors_[idx].target_fps;
                double process_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_start).count();
                if (process_ms < target_ms) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(target_ms - process_ms)));
                }
            }
            pclose(pipe);
        } else {
            // 2. USB 카메라 처리 방식: V4L2
            cv::VideoCapture cap;
            
            // "auto_usb" 모드일 경우 0~10번 노드를 순회하며 유효한 카메라 탐색
            if (source == "auto_usb") {
                for (int i = 0; i <= 10; ++i) {
                    std::string dev_name = "/dev/video" + std::to_string(i);
                    cap.open(dev_name, cv::CAP_V4L2);
                    
                    if (cap.isOpened()) {
                        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                        cap.set(cv::CAP_PROP_FPS, 120);

                        cv::Mat test_frame;
                        if (cap.read(test_frame) && !test_frame.empty()) {
                            RCLCPP_INFO(this->get_logger(), "Auto-detected valid USB camera at %s", dev_name.c_str());
                            break; // 성공하면 루프 탈출 (현재 cap 유지)
                        }
                        cap.release(); // 프레임을 못 읽으면 가짜 노드이므로 닫고 다음 노드로
                    }
                }
            } 
            // 수동 지정 (예: "/dev/video2")일 경우 그대로 열기
            else {
                cap.open(source, cv::CAP_V4L2);
                
                if (cap.isOpened()) {
                    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                    cap.set(cv::CAP_PROP_FPS, 120);
                }
            }

            if (!cap.isOpened()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to find or open any USB camera (source: %s)", source.c_str());
                return; // 스레드 종료
            }

            cv::Mat frame;
            auto last_frame_time = std::chrono::steady_clock::now();
            int frame_count = 0;

            // 영상 처리 루프 (기존 코드와 동일)
            while (running_ && rclcpp::ok()) {
                auto loop_start = std::chrono::steady_clock::now();
                if (!cap.read(frame)) continue;
                if (frame.channels() == 3) cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
                process_frame(frame);

                frame_count++;
                auto now = std::chrono::steady_clock::now();
                double elapsed_sec = std::chrono::duration<double>(now - last_frame_time).count();
                if (elapsed_sec >= 1.0) {
                    sensors_[idx].current_fps = frame_count / elapsed_sec;
                    frame_count = 0;
                    last_frame_time = now;
                }

                double target_ms = 1000.0 / sensors_[idx].target_fps;
                double process_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_start).count();
                if (process_ms < target_ms) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(target_ms - process_ms)));
                }
            }
        }
    }

    void calculate_and_publish() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        double f1 = sensors_[0].force;
        double f2 = sensors_[1].force;
        double f3 = sensors_[2].force;
        double f_total = f1 + f2 + f3;

        rclcpp::Time current_time = this->now();
        double dt = (current_time - last_time_).seconds();
        if (dt <= 0) dt = 0.01;

        double ball_x = 0.0;
        double ball_y = 0.0;

        if (f_total > 0.01) { 
            double pos1_x = eef_ptr_[0], pos1_y = eef_ptr_[1];
            double pos2_x = eef_ptr_[2], pos2_y = eef_ptr_[3];
            double pos3_x = eef_ptr_[4], pos3_y = eef_ptr_[5];

            ball_x = (pos1_x * f1 + pos2_x * f2 + pos3_x * f3) / f_total;
            ball_y = (pos1_y * f1 + pos2_y * f2 + pos3_y * f3) / f_total;
        }

        double vx = (ball_x - last_ball_x_) / dt;
        double vy = (ball_y - last_ball_y_) / dt;

        shm_ptr_[0] = ball_x;
        shm_ptr_[1] = ball_y;
        shm_ptr_[2] = vx;
        shm_ptr_[3] = vy;
        shm_ptr_[4] = current_time.seconds();

        last_ball_x_ = ball_x;
        last_ball_y_ = ball_y;
        last_time_ = current_time;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TactileBallTracker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}