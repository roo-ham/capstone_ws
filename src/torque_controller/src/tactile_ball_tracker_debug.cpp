#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
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
#include <algorithm> 
#include <std_msgs/msg/empty.hpp>

using json = nlohmann::json;
using namespace std::chrono_literals;

struct SensorData {
    double force = 0.0;
    double last_area = 0.0;
    double zero_area = 0.0;
    
    double k_val = 100.0; 
    double b_val = 100.0; 
    int target_fps = 60;
    
    std::vector<cv::Point> trap_pts; 
    cv::Mat mask;                    
    cv::UMat u_mask;                    

    int thresh = 120;
    double current_fps = 0.0;
    
    cv::Mat display_img;
};

class TactileBallTracker : public rclcpp::Node {
public:
    TactileBallTracker() : Node("tactile_ball_tracker") {
        last_time_ = this->now(); 
        
        load_config();
        init_shm();
        
        this->declare_parameter("fps_limit", 60);
        this->declare_parameter("filter_alpha", 1.0);
        this->declare_parameter("show_gui", true);

        zero_sub_ = this->create_subscription<std_msgs::msg::Empty>(
                "set_force_zero", 10,
                [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
            std::lock_guard<std::mutex> lock(data_mutex_);

            for (int i = 0; i < 3; ++i) {
                sensors_[i].b_val = -sensors_[i].last_area;
                // b_val만 업데이트, threshold는 고정 120
            }

            // Save b_val to JSON (persist across restarts)
            {
                std::ifstream file("/home/kimdonghwi/capstone_ws_claude/tactile_config.json");
                json j;
                if (file.is_open()) { file >> j; file.close(); }
                if (j.contains("cameras")) {
                    for (int i = 0; i < 3; ++i) {
                        if (j["cameras"].size() > (size_t)i) {
                            j["cameras"][i]["b_slider"] = sensors_[i].b_val;
                        }
                    }
                    std::ofstream out("/home/kimdonghwi/capstone_ws_claude/tactile_config.json");
                    if (out.is_open()) { out << j.dump(4); out.close(); }
                }
            }
            RCLCPP_INFO(this->get_logger(), "Set Force Zero: b_val updated, ref intact. Saved to JSON.");
        });

        for (int i = 0; i < 3; ++i) {
            running_ = true;
            capture_threads_.emplace_back(&TactileBallTracker::camera_thread_func, this, i);
        }

        timer_ = this->create_wall_timer(
            5ms, std::bind(&TactileBallTracker::calculate_and_publish, this));

        ui_timer_ = this->create_wall_timer(
            100ms, std::bind(&TactileBallTracker::update_terminal_ui, this));
        
        cv::ocl::setUseOpenCL(true);
        RCLCPP_INFO(this->get_logger(), "OpenCL Acceleration: %s", 
                    cv::ocl::useOpenCL() ? "ENABLED" : "DISABLED");
        RCLCPP_INFO(this->get_logger(), "Optimized Tactile Ball Tracker Node Started.");
    }

    ~TactileBallTracker() {
        running_ = false;
        for (auto& th : capture_threads_) {
            if (th.joinable()) th.join();
        }

        munmap(shm_ptr_, 7 * sizeof(double));
        close(fd_shm_);
        shm_unlink("ball_state_shm");

        if (pose_ptr_ && pose_ptr_ != MAP_FAILED) {
            munmap(pose_ptr_, 12 * sizeof(double));
        }
        if (fd_pose_ != -1) {
            close(fd_pose_);
        }

        if (show_gui_) cv::destroyAllWindows();
    }

private:
    double* shm_ptr_; 
    double* pose_ptr_ = nullptr; 
    int fd_shm_, fd_pose_ = -1;

    std::vector<std::thread> capture_threads_;
    bool running_;
    SensorData sensors_[3];
    std::string camera_sources_[3];
    std::string rpicam_cmds_[3]; 
    
    std::mutex data_mutex_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr ui_timer_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr zero_sub_;
    bool show_gui_ = true;

    rclcpp::Time last_time_;
    double prev_f1 = 0.0, prev_f2 = 0.0, prev_f3 = 0.0;
    double prev_force_time_ = 0.0;

    void init_shm() {
        shm_unlink("ball_state_shm");
        fd_shm_ = shm_open("ball_state_shm", O_CREAT | O_RDWR, 0666);
        
        ftruncate(fd_shm_, 7 * sizeof(double));
        shm_ptr_ = (double*)mmap(0, 7 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm_, 0);
        std::fill(shm_ptr_, shm_ptr_ + 7, 0.0);

        connect_pose_shm();
    }

    void connect_pose_shm() {
        if (pose_ptr_ == nullptr) {
            fd_pose_ = shm_open("target_pose_shm", O_RDONLY, 0666);
            if (fd_pose_ != -1) {
                void* ptr = mmap(0, 12 * sizeof(double), PROT_READ, MAP_SHARED, fd_pose_, 0);
                if (ptr != MAP_FAILED) {
                    pose_ptr_ = (double*)ptr;
                    RCLCPP_INFO(this->get_logger(), "Successfully connected to target_pose_shm.");
                } else {
                    close(fd_pose_);
                    fd_pose_ = -1;
                }
            }
        }
    }

    void load_config() {
        std::ifstream file("/home/kimdonghwi/capstone_ws_claude/tactile_config.json");
        if (file.is_open()) {
            json j;
            file >> j;
            for (int i = 0; i < 3; ++i) {
                camera_sources_[i] = j["cameras"][i]["source"].get<std::string>();
                
                if (j["cameras"][i].contains("rpicam_cmd")) {
                    rpicam_cmds_[i] = j["cameras"][i]["rpicam_cmd"].get<std::string>();
                } else {
                    rpicam_cmds_[i] = "";
                }

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
        int global_fps_limit = this->get_parameter("fps_limit").as_int();
        show_gui_ = this->get_parameter("show_gui").as_bool();
        for(int i=0; i<3; ++i) sensors_[i].target_fps = global_fps_limit;

        std::vector<cv::Mat> disp_imgs(3);

        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            printf("\033[H\033[J");
            printf("====================================================\n");
            printf("         OPTIMIZED TACTILE SENSOR TRACKER TUI       \n");
            printf("====================================================\n");
            printf(" Target FPS Limit: %d (Adjust via ROS Param)\n\n", global_fps_limit);
            
            for(int i=0; i<3; ++i) {
                printf(" [Sensor %d] FPS: %5.1f | Area: %7.1f | Force: %6.2f (dF: %6.1f) | Thresh: %3d\n",
                       i+1, sensors_[i].current_fps, sensors_[i].last_area, sensors_[i].force,
                       shm_ptr_[3+i], sensors_[i].thresh);
                
                if(!sensors_[i].display_img.empty()) {
                    disp_imgs[i] = sensors_[i].display_img.clone();
                }
            }
            printf("====================================================\n");
        }
        fflush(stdout);

        if (show_gui_) {
            for(int i=0; i<3; ++i) {
                if(!disp_imgs[i].empty()) {
                    cv::imshow("Camera " + std::to_string(i) + " Binary", disp_imgs[i]);
                }
            }
            cv::waitKey(1);
        }
    }

    void camera_thread_func(int idx) {
        std::string source = camera_sources_[idx];
        
        auto process_frame = [&](cv::Mat& frame) {
            if (frame.empty()) return;

            cv::UMat u_frame, u_binary;

            try {
                frame.copyTo(u_frame); 

                if (sensors_[idx].u_mask.empty() && !sensors_[idx].trap_pts.empty()) {
                    cv::Mat temp_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                    cv::fillConvexPoly(temp_mask, sensors_[idx].trap_pts, cv::Scalar(255));
                    temp_mask.copyTo(sensors_[idx].u_mask);
                }

                cv::threshold(u_frame, u_binary, 120, 255, cv::THRESH_BINARY);
                
                if (!sensors_[idx].u_mask.empty()) {
                    cv::bitwise_and(u_binary, sensors_[idx].u_mask, u_binary);
                }

                cv::Mat display_clone;
                u_binary.copyTo(display_clone); 

                double total_area = cv::countNonZero(u_binary);
                double alpha = this->get_parameter("filter_alpha").as_double();

                {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    
                    double smoothed_area = alpha * total_area + (1.0 - alpha) * sensors_[idx].last_area;
                    sensors_[idx].display_img = display_clone; 
                    
                    double k = sensors_[idx].k_val;
                    double b = sensors_[idx].b_val;

                    sensors_[idx].last_area = smoothed_area;
                    sensors_[idx].force = (k * (smoothed_area+b));
                } 

                u_frame.release();
                u_binary.release();

            } catch (const cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "[Sensor %d] OpenCV Exception in process_frame: %s", idx + 1, e.what());
                u_frame.release();
                u_binary.release();
                sensors_[idx].u_mask.release(); 
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "[Sensor %d] Standard Exception in process_frame: %s", idx + 1, e.what());
            } catch (...) {
                RCLCPP_ERROR(this->get_logger(), "[Sensor %d] Unknown Exception occurred in process_frame.", idx + 1);
            }
        };

        if (source == "0" || source == "1") {
            std::string cmd = rpicam_cmds_[idx];
            if (cmd.empty()) {
                cmd = "rpicam-vid -t 0 --camera " + source + 
                      " --width 640 --height 400 --framerate 120" + 
                      " --codec yuv420 --denoise off --nopreview --awb daylight --shutter 20000 --gain 2.0 -o - 2>/dev/null";
            }
            
            FILE* pipe = popen(cmd.c_str(), "r");
            if (!pipe) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open rpicam pipe for sensor %d", idx+1);
                return;
            }

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
            pclose(pipe);
        } else {
            cv::VideoCapture cap;
            std::string dev_name;

            if (source.find("/dev/") == 0) {
                dev_name = source; 
            } else {
                dev_name = "/dev/video" + source; 
            }

            // [수정 1] 카메라를 먼저 엽니다. (OpenCV 초기화가 시스템 설정을 덮어쓰는 것을 방지)
            cap.open(dev_name, cv::CAP_V4L2);
            if (cap.isOpened()) {
                cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                cap.set(cv::CAP_PROP_FPS, 120);

                // OpenCV 내부 API로도 1차 고정 (V4L2 백엔드 기준 1이 수동)
                cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1); 
                cap.set(cv::CAP_PROP_EXPOSURE, 150);
                cap.set(cv::CAP_PROP_GAIN, 20);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to find or open any USB camera (source: %s)", source.c_str());
                return;
            }

            // [수정 2] 하드웨어 파라미터를 강제하는 람다 함수 정의
            auto apply_camera_settings = [dev_name]() {
                // 터미널 출력을 막기 위해 > /dev/null 2>&1 추가
                std::string suffix = " > /dev/null 2>&1";
                
                // 기존 설정
                system(("v4l2-ctl -d " + dev_name + " -c auto_exposure=1" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c exposure_time_absolute=150" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c gain=20" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c white_balance_automatic=0" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c white_balance_temperature=4600" + suffix).c_str());
                
                // 드리프팅을 유발하는 숨겨진 ISP 기능들 모두 Off
                system(("v4l2-ctl -d " + dev_name + " -c backlight_compensation=0" + suffix).c_str()); // 역광 보정 끄기
                system(("v4l2-ctl -d " + dev_name + " -c power_line_frequency=0" + suffix).c_str());   // 플리커 방지(50/60Hz) 자동조절 끄기
                system(("v4l2-ctl -d " + dev_name + " -c exposure_dynamic_framerate=0" + suffix).c_str()); // 동적 프레임레이트 끄기
            };

            // cap.open() 직후에 V4L2 명령으로 하드웨어 세팅 덮어쓰기
            apply_camera_settings();

            cv::Mat frame;
            auto last_frame_time = std::chrono::steady_clock::now();
            int frame_count = 0;

            while (running_ && rclcpp::ok()) {
                auto loop_start = std::chrono::steady_clock::now();
                if (!cap.read(frame)) continue;
                if (frame.channels() == 3) cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
                process_frame(frame);

                frame_count++;
                auto now = std::chrono::steady_clock::now();
                
                // FPS 계산 로직
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

        connect_pose_shm();

        double f1 = (sensors_[0].force);
        double f2 = (sensors_[1].force);
        double f3 = (sensors_[2].force);

        rclcpp::Time current_time = this->now();
        double t_now = current_time.seconds();
        double force_dt = (prev_force_time_ > 0.0) ? (t_now - prev_force_time_) : (1.0 / 60.0);
        if (force_dt <= 0.0) force_dt = 1.0 / 60.0;

        double df1 = (f1 - prev_f1) / force_dt;
        double df2 = (f2 - prev_f2) / force_dt;
        double df3 = (f3 - prev_f3) / force_dt;

        prev_f1 = f1; prev_f2 = f2; prev_f3 = f3;
        prev_force_time_ = t_now;

        shm_ptr_[0] = f1;
        shm_ptr_[1] = f2;
        shm_ptr_[2] = f3;
        shm_ptr_[3] = df1;
        shm_ptr_[4] = df2;
        shm_ptr_[5] = df3;
        shm_ptr_[6] = current_time.seconds();

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