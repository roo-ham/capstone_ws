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
#include <numeric>
#include <algorithm>
#include <std_msgs/msg/empty.hpp>

// Qt5 Headers
#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QGridLayout>
#include <QLabel>
#include <QVBoxLayout>
#include <QMessageBox>

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

    int thresh = 50;
    double current_fps = 0.0;
};

struct RecordData {
    double area[3];
    double target_x;
    double target_y;
};

class TactileBallTrackerDebug : public rclcpp::Node {
public:
    TactileBallTrackerDebug() : Node("tactile_ball_tracker_debug") {
        last_time_ = this->now(); 
        
        load_config();
        init_shm();
        
        this->declare_parameter("fps_limit", 60);
        this->declare_parameter("filter_alpha", 1.0); 

        running_ = true;
        for (int i = 0; i < 3; ++i) {
            capture_threads_.emplace_back(&TactileBallTrackerDebug::camera_thread_func, this, i);
        }

        timer_ = this->create_wall_timer(
            10ms, std::bind(&TactileBallTrackerDebug::calculate_and_record, this));

        cv::ocl::setUseOpenCL(true);
        RCLCPP_INFO(this->get_logger(), "OpenCL Acceleration: %s", 
                    cv::ocl::useOpenCL() ? "ENABLED" : "DISABLED");
        RCLCPP_INFO(this->get_logger(), "Debug Node Started with Qt GUI.");
    }

    ~TactileBallTrackerDebug() {
        running_ = false;
        for (auto& th : capture_threads_) {
            if (th.joinable()) th.join();
        }
        
        if (eef_ptr_) munmap(eef_ptr_, 6 * sizeof(double));
        close(fd_eef_);
        if (shm_ptr_) munmap(shm_ptr_, 5 * sizeof(double));
        close(fd_shm_);
        shm_unlink("ball_state_shm");
    }

    void set_force_zero() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        for (int i = 0; i < 3; ++i) {
            sensors_[i].b_val = -sensors_[i].last_area;
        }
        RCLCPP_INFO(this->get_logger(), "Zero force offset applied.");
    }

    void toggle_recording(int grid_id, double tx, double ty, bool start) {
        std::lock_guard<std::mutex> lock(record_mutex_);
        if (start) {
            is_recording_ = true;
            current_grid_id_ = grid_id;
            target_x_ = tx;
            target_y_ = ty;
            temp_buffer_.clear();
            RCLCPP_INFO(this->get_logger(), "Started recording grid %d (%.2f, %.2f)", grid_id, tx, ty);
        } else {
            is_recording_ = false;
            if (!temp_buffer_.empty()) {
                double avg_a0 = 0, avg_a1 = 0, avg_a2 = 0;
                for (const auto& val : temp_buffer_) {
                    avg_a0 += val[0]; avg_a1 += val[1]; avg_a2 += val[2];
                }
                size_t n = temp_buffer_.size();
                avg_a0 /= n; avg_a1 /= n; avg_a2 /= n;

                RecordData rd = {{avg_a0, avg_a1, avg_a2}, tx, ty};
                recorded_data_[grid_id] = rd;
                RCLCPP_INFO(this->get_logger(), "Stopped. Saved %zu samples for grid %d.", n, grid_id);
            }
        }
    }

    std::string calculate_fit() {
        std::lock_guard<std::mutex> lock(record_mutex_);
        if (recorded_data_.empty()) return "Error: No data recorded yet.";
        if (!eef_ptr_) return "Error: SHM eef_ptr_ is not initialized.";

        double pos1_x = eef_ptr_[0], pos1_y = eef_ptr_[1];
        double pos2_x = eef_ptr_[2], pos2_y = eef_ptr_[3];
        double pos3_x = eef_ptr_[4], pos3_y = eef_ptr_[5];

        double det = (pos2_x * pos3_y - pos3_x * pos2_y) 
                   - (pos1_x * pos3_y - pos3_x * pos1_y) 
                   + (pos1_x * pos2_y - pos2_x * pos1_y);

        if (std::abs(det) < 1e-6) return "Error: Sensor positions are invalid or collinear.";

        std::vector<double> areas[3];
        std::vector<double> forces[3];

        // [추가] CSV 저장을 위한 파일 스트림 열기
        std::ofstream csv_file("calibration_data.csv");
        if (csv_file.is_open()) {
            csv_file << "Grid_X,Grid_Y,Real_Area_S1,Real_Area_S2,Real_Area_S3,Target_F1(gf),Target_F2(gf),Target_F3(gf)\n";
        }

        for (const auto& pair : recorded_data_) {
            const RecordData& rd = pair.second;
            
            double F_tot = 140.0;
            double Mx = F_tot * rd.target_x;
            double My = F_tot * rd.target_y;

            double det1 = F_tot * (pos2_x * pos3_y - pos3_x * pos2_y) - 1 * (Mx * pos3_y - pos3_x * My) + 1 * (Mx * pos2_y - pos2_x * My);
            double det2 = 1 * (Mx * pos3_y - pos3_x * My) - F_tot * (pos1_x * pos3_y - pos3_x * pos1_y) + 1 * (pos1_x * My - Mx * pos1_y);
            double det3 = 1 * (pos2_x * My - Mx * pos2_y) - 1 * (pos1_x * My - Mx * pos1_y) + F_tot * (pos1_x * pos2_y - pos2_x * pos1_y);

            // RCLCPP_INFO(this->get_logger(), "\n[EEF Sensor Coordinates Check]");
            // RCLCPP_INFO(this->get_logger(), "S1 (X, Y) = (%.4f, %.4f)", pos1_x, pos1_y);
            // RCLCPP_INFO(this->get_logger(), "S2 (X, Y) = (%.4f, %.4f)", pos2_x, pos2_y);
            // RCLCPP_INFO(this->get_logger(), "S3 (X, Y) = (%.4f, %.4f)", pos3_x, pos3_y);

            double target_f1 = det1 / det;
            double target_f2 = det2 / det;
            double target_f3 = det3 / det;

            areas[0].push_back(rd.area[0]); forces[0].push_back(target_f1);
            areas[1].push_back(rd.area[1]); forces[1].push_back(target_f2);
            areas[2].push_back(rd.area[2]); forces[2].push_back(target_f3);

            // [추가] 매칭된 데이터를 CSV에 행(Row) 단위로 기록
            if (csv_file.is_open()) {
                csv_file << rd.target_x << "," << rd.target_y << ","
                         << rd.area[0] << "," << rd.area[1] << "," << rd.area[2] << ","
                         << target_f1 << "," << target_f2 << "," << target_f3 << "\n";
            }
        }
        
        if (csv_file.is_open()) {
            csv_file.close();
            RCLCPP_INFO(this->get_logger(), "Calibration data saved to calibration_data.csv");
        }

        std::stringstream ss;
        ss.precision(4);

        for (int i = 0; i < 3; ++i) {
            double n = areas[i].size();
            if (n < 2) return "Error: Need at least 2 distinct grid points recorded.";

            double sum_A = std::accumulate(areas[i].begin(), areas[i].end(), 0.0);
            double sum_F = std::accumulate(forces[i].begin(), forces[i].end(), 0.0);
            
            double mean_A = sum_A / n;
            double mean_F = sum_F / n;

            double S_AA = 0, S_AF = 0;
            for (size_t j = 0; j < n; ++j) {
                S_AA += (areas[i][j] - mean_A) * (areas[i][j] - mean_A);
                S_AF += (areas[i][j] - mean_A) * (forces[i][j] - mean_F);
            }

            if (std::abs(S_AA) < 1e-6) S_AA = 1e-6; // 나눗셈 제로 방지

            double k_fit = S_AF / S_AA;
            double c_fit = mean_F - k_fit * mean_A; 
            double b_fit = (std::abs(k_fit) > 1e-6) ? (c_fit / k_fit) : 0.0;

            double sum_sq_err = 0;
            for (size_t j = 0; j < n; ++j) {
                double pred = k_fit * (areas[i][j] + b_fit);
                sum_sq_err += (forces[i][j] - pred) * (forces[i][j] - pred);
            }
            double std_dev = std::sqrt(sum_sq_err / n);

            ss << "[Sensor " << i+1 << "]\n"
               << "  k = " << k_fit << "\n"
               << "  b = " << b_fit << "\n"
               << "  StdDev(gf) = " << std_dev << "\n\n";
        }

        // GUI 상에도 CSV 저장 완료 메시지 출력
        ss << "-> Saved raw data to 'calibration_data.csv'";

        return ss.str();
    }

private:
    double* shm_ptr_ = nullptr; 
    double* eef_ptr_ = nullptr; 
    int fd_shm_ = -1, fd_eef_ = -1;

    std::vector<std::thread> capture_threads_;
    bool running_ = false;
    SensorData sensors_[3];
    std::string camera_sources_[3];
    
    std::mutex data_mutex_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time last_time_;

    std::mutex record_mutex_;
    bool is_recording_ = false;
    int current_grid_id_ = -1;
    double target_x_ = 0.0, target_y_ = 0.0;
    std::vector<std::array<double, 3>> temp_buffer_;
    std::map<int, RecordData> recorded_data_;

    void init_shm() {
        fd_eef_ = shm_open("eef_pos_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(fd_eef_, 6 * sizeof(double));
        eef_ptr_ = (double*)mmap(0, 6 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_eef_, 0);

        shm_unlink("ball_state_shm");
        fd_shm_ = shm_open("ball_state_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(fd_shm_, 5 * sizeof(double));
        shm_ptr_ = (double*)mmap(0, 5 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm_, 0);
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
                
                sensors_[i].trap_pts.clear();
                for(auto& pt : j["cameras"][i]["trap_src"]) {
                    sensors_[i].trap_pts.push_back(cv::Point(pt[0], pt[1]));
                }
            }
            file.close();
        }
    }

    void camera_thread_func(int idx) {
        std::string source = camera_sources_[idx];
        
        auto process_frame = [&](cv::Mat& frame) {
            cv::UMat u_frame, u_binary;
            
            // 1. CPU(frame) -> GPU(u_frame) 메모리 복사
            frame.copyTo(u_frame); 

            // 2. 마스크 최초 1회 생성 및 GPU 메모리(u_mask)에 캐싱
            if (sensors_[idx].u_mask.empty() && !sensors_[idx].trap_pts.empty()) {
                cv::Mat temp_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                cv::fillConvexPoly(temp_mask, sensors_[idx].trap_pts, cv::Scalar(255));
                temp_mask.copyTo(sensors_[idx].u_mask); // GPU로 올려둠
            }

            // 3. iGPU를 이용한 초고속 이진화 및 마스킹 (T-API가 알아서 GPU 연산 수행)
            cv::threshold(u_frame, u_binary, sensors_[idx].thresh, 255, cv::THRESH_BINARY);
            
            if (!sensors_[idx].u_mask.empty()) {
                cv::bitwise_and(u_binary, sensors_[idx].u_mask, u_binary);
            }

            // 5. 픽셀 개수 세기 (이것도 GPU에서 연산 후 결과값만 CPU로 반환)
            double total_area = cv::countNonZero(u_binary);
            
            double alpha = this->get_parameter("filter_alpha").as_double();
            double smoothed_area = alpha * total_area + (1.0 - alpha) * sensors_[idx].last_area;
            
            std::lock_guard<std::mutex> lock(data_mutex_);
            sensors_[idx].last_area = smoothed_area;
            sensors_[idx].force = (sensors_[idx].k_val * (smoothed_area + sensors_[idx].b_val));
        };

        // 원본 노드의 카메라 백엔드 로직 완벽 복구
        if (source == "0" || source == "1") {
            std::string cmd = "rpicam-vid -t 0 --camera " + source + 
                              " --width 640 --height 400 --framerate 120" + 
                              " --codec yuv420 --denoise off --nopreview -o - 2>/dev/null";
            
            FILE* pipe = popen(cmd.c_str(), "r");
            if (!pipe) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open rpicam pipe for camera %s", source.c_str());
                return;
            }

            int width = 640, height = 400;
            int y_size = width * height;           
            int frame_size = y_size * 3 / 2;       
            
            std::vector<uint8_t> buffer(frame_size);
            cv::Mat frame(height, width, CV_8UC1); 

            while (running_ && rclcpp::ok()) {
                size_t bytes_read = fread(buffer.data(), 1, frame_size, pipe);
                if (bytes_read != (size_t)frame_size) continue;

                std::memcpy(frame.data, buffer.data(), y_size);
                process_frame(frame);
            }
            pclose(pipe);
        } 
        else {
            cv::VideoCapture cap;
            if (source == "auto_usb") {
                for (int i = 0; i <= 10; ++i) {
                    std::string dev_name = "/dev/video" + std::to_string(i);
                    cap.open(dev_name, cv::CAP_V4L2);
                    
                    if (cap.isOpened()) {
                        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                        cap.set(cv::CAP_PROP_FPS, 120);

                        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1); 

                        // 2. 화이트 밸런스 자동 기능 비활성화 (밝기/색감 변화 원인 차단)
                        cap.set(cv::CAP_PROP_AUTO_WB, 0);

                        // 3. 수동 노출값 세팅 (예: 150)
                        // ※ 주의: 카메라 모듈마다 받아들이는 수치의 범위(예: 1~5000 또는 -1~-11)가 다릅니다.
                        // 터미널에서 v4l2-ctl -L로 범위를 먼저 확인한 후 적절한 값을 기입하세요.
                        cap.set(cv::CAP_PROP_EXPOSURE, 150);

                        cv::Mat test_frame;
                        if (cap.read(test_frame) && !test_frame.empty()) {
                            RCLCPP_INFO(this->get_logger(), "Auto-detected valid USB camera at %s", dev_name.c_str());
                            break; 
                        }
                        cap.release(); 
                    }
                }
            } else {
                cap.open(source, cv::CAP_V4L2);
                if (cap.isOpened()) {
                    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                    cap.set(cv::CAP_PROP_FPS, 120);
                }
            }

            if (!cap.isOpened()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open USB camera (source: %s)", source.c_str());
                return;
            }

            cv::Mat frame;
            while (running_ && rclcpp::ok()) {
                if (!cap.read(frame)) continue;
                if (frame.empty()) continue;
                if (frame.channels() == 3) cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
                process_frame(frame);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    }

    void calculate_and_record() {
        double a1, a2, a3;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            a1 = sensors_[0].last_area;
            a2 = sensors_[1].last_area;
            a3 = sensors_[2].last_area;
        }

        {
            std::lock_guard<std::mutex> rec_lock(record_mutex_);
            if (is_recording_) {
                temp_buffer_.push_back({a1, a2, a3});
            }
        }
    }
};

// Qt GUI 클래스 동일
class DebugUI : public QWidget {
public:
    DebugUI(std::shared_ptr<TactileBallTrackerDebug> node) : node_(node) {
        setWindowTitle("Tactile Sensor Calibration UI");
        setFixedSize(500, 600);

        QVBoxLayout* main_layout = new QVBoxLayout(this);

        QPushButton* zero_btn = new QPushButton("Set Force Zero (Offset)");
        zero_btn->setStyleSheet("background-color: lightgray; padding: 10px; font-weight: bold;");
        connect(zero_btn, &QPushButton::clicked, [this]() {
            node_->set_force_zero();
        });
        main_layout->addWidget(zero_btn);

        QGridLayout* grid_layout = new QGridLayout();
        double coords[3] = {-0.15, 0.0, 0.15};
        
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                int id = r * 3 + c;
                double target_x = coords[c];
                double target_y = -coords[r]; 

                grid_btns_[id] = new QPushButton(QString("(%1, %2)").arg(target_x).arg(target_y));
                grid_btns_[id]->setCheckable(true);
                grid_btns_[id]->setMinimumHeight(60);
                grid_btns_[id]->setStyleSheet("QPushButton { background-color: #E0E0E0; }"
                                              "QPushButton:checked { background-color: #FF6666; font-weight: bold; }");
                
                connect(grid_btns_[id], &QPushButton::toggled, [this, id, target_x, target_y](bool checked) {
                    node_->toggle_recording(id, target_x, target_y, checked);
                });

                grid_layout->addWidget(grid_btns_[id], r, c);
            }
        }
        main_layout->addLayout(grid_layout);

        QPushButton* fit_btn = new QPushButton("Calculate K, B Fit (Least Squares)");
        fit_btn->setStyleSheet("background-color: lightgreen; padding: 10px; font-weight: bold;");
        main_layout->addWidget(fit_btn);

        result_label_ = new QLabel("Fit results will be displayed here.\nPress buttons to collect 140gf data.");
        result_label_->setFrameStyle(QFrame::Panel | QFrame::Sunken);
        result_label_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        main_layout->addWidget(result_label_);

        connect(fit_btn, &QPushButton::clicked, [this]() {
            std::string result = node_->calculate_fit();
            result_label_->setText(QString::fromStdString(result));
        });
    }

private:
    std::shared_ptr<TactileBallTrackerDebug> node_;
    QPushButton* grid_btns_[9];
    QLabel* result_label_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TactileBallTrackerDebug>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread ros_thread([&executor]() {
        executor.spin();
    });

    QApplication app(argc, argv);
    DebugUI ui(node);
    ui.show();
    int ret = app.exec();

    rclcpp::shutdown();
    if (ros_thread.joinable()) {
        ros_thread.join();
    }
    
    return ret;
}