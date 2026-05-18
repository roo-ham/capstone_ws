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
#include <vector>
#include <iomanip>
#include <cstring> // std::memcpy 사용을 위해 추가
#include <Eigen/Dense>

// Qt5 Headers
#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QGridLayout>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTextEdit>
#include <QMessageBox>
#include <QGroupBox>
#include <QTimer>
#include <QMetaObject>

using json = nlohmann::json;

struct SensorData {
    double b_val = 0.0;
    double k_val = 0.0;
    std::string source;
    std::string rpicam_cmd;
    std::vector<cv::Point> trap_pts;
    double current_area = 0.0;

    cv::VideoCapture cap;
    cv::Mat mask;

    // Adaptive threshold (same as tactile_ball_tracker)
    cv::Rect ref_roi = cv::Rect(5, 5, 30, 30);
    double ref_brightness = 0.0;
    double ref_baseline = -1.0;
    int warmup_count = 0;
    static constexpr int WARMUP_FRAMES = 200;
    int thresh = 120;
    int base_thresh = 120;

    // 실시간 상태 변수
    double current_brightness = 0.0;
    bool is_running = false;
};

// 측정 데이터를 저장할 구조체 (물리량 + 이미지 면적 + EEF 실제 위치)
struct MeasureData {
    std::string label;
    double m_total; // 총 질량 (M_total)
    double m_add;   // 추가된 질량 (m)
    double pos_x;   // 올려진 물체의 x 좌표
    double pos_y;   // 올려진 물체의 y 좌표
    double area[3]; // 센서 1,2,3 의 white pixel 면적
    double eef_x[3]; // 측정 당시 손가락 1,2,3 의 실제 X 좌표
    double eef_y[3]; // 측정 당시 손가락 1,2,3 의 실제 Y 좌표
};

class BallCalibrationNode : public rclcpp::Node {
public:
    BallCalibrationNode() : Node("ball_calibration_node"), running_(true), is_paused_(true), shm_connected_(false) {
        
        // 1. target_data 초기화
        double defaults[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 10.0, 1.0, 0.2, 0.2, 50.0, 5.0};
        for(int i=0; i<12; ++i) target_data_[i] = defaults[i];

        // 2. Config 자동 로드
        load_tactile_config("/home/kimdonghwi/capstone_ws_claude/tactile_config.json");
        load_balance_config("/home/kimdonghwi/capstone_ws_claude/balance_config.json");

        // 3. 백그라운드 SHM 연결 및 비전 처리 스레드 시작
        shm_thread_ = std::thread(&BallCalibrationNode::shm_loop, this);
        for (int i = 0; i < 3; ++i) {
            capture_threads_.emplace_back(&BallCalibrationNode::camera_loop, this, i);
        }
    }

    ~BallCalibrationNode() {
        running_ = false;
        if (shm_thread_.joinable()) shm_thread_.join();
        for (auto& th : capture_threads_) {
            if (th.joinable()) th.join();
        }
        if (pose_ptr_ && pose_ptr_ != MAP_FAILED) munmap(pose_ptr_, 12 * sizeof(double));
        if (eef_ptr_ && eef_ptr_ != MAP_FAILED) munmap(eef_ptr_, 9 * sizeof(double));
        if (fd_pose_ != -1) close(fd_pose_);
        if (fd_eef_ != -1) close(fd_eef_);
    }

    // --- JSON Load / Save ---
    void load_tactile_config(const std::string& filename) {
        tactile_config_file_ = filename;
        std::ifstream file(filename);
        if (!file.is_open()) return;
        json j; file >> j;
        
        sensors_.resize(3);
        for (size_t i = 0; i < 3 && i < j["cameras"].size(); ++i) {
            auto& cam = j["cameras"][i];
            sensors_[i].b_val = cam.value("b_slider", 0.0);
            sensors_[i].k_val = cam.value("k_slider", 1.0);
            sensors_[i].source = cam.value("source", "0");
            sensors_[i].rpicam_cmd = cam.value("rpicam_cmd", "");
            
            sensors_[i].trap_pts.clear();
            for (auto& pt : cam["trap_src"]) {
                sensors_[i].trap_pts.push_back(cv::Point(pt[0], pt[1]));
            }
            // reference ROI for adaptive threshold
            if (cam.contains("ref_roi")) {
                auto& rr = cam["ref_roi"];
                sensors_[i].ref_roi = cv::Rect(rr[0], rr[1], rr[2], rr[3]);
            }
            // restore adaptive threshold state
            if (cam.contains("ref_baseline")) {
                sensors_[i].ref_baseline = cam["ref_baseline"];
                sensors_[i].ref_brightness = cam.value("ref_brightness", sensors_[i].ref_baseline);
                sensors_[i].warmup_count = sensors_[i].WARMUP_FRAMES;
            }

            sensors_[i].mask = cv::Mat::zeros(400, 640, CV_8UC1);
            std::vector<std::vector<cv::Point>> pts = {sensors_[i].trap_pts};
            cv::fillPoly(sensors_[i].mask, pts, cv::Scalar(255));
        }
    }

    void save_tactile_config() {
        std::ifstream file(tactile_config_file_);
        json j;
        if (file.is_open()) {
            file >> j;
            file.close();
        } else {
            RCLCPP_ERROR(this->get_logger(), "Cannot read %s for saving.", tactile_config_file_.c_str());
            return;
        }
        if (!j.contains("cameras")) {
            RCLCPP_ERROR(this->get_logger(), "Invalid JSON in %s, refusing to overwrite.", tactile_config_file_.c_str());
            return;
        }
        for (int i = 0; i < 3; ++i) {
            j["cameras"][i]["k_slider"] = sensors_[i].k_val;
            j["cameras"][i]["b_slider"] = sensors_[i].b_val;
            j["cameras"][i]["ref_baseline"] = sensors_[i].ref_baseline;
            j["cameras"][i]["ref_brightness"] = sensors_[i].ref_brightness;
        }
        std::ofstream out(tactile_config_file_);
        out << j.dump(4);
    }

    void load_balance_config(const std::string& filename) {
        balance_config_file_ = filename;
        std::ifstream file(filename);
        if (file.is_open()) {
            file >> balance_json_;
            for (int i = 5; i < 12; ++i) {
                std::string idx_str = std::to_string(i);
                if (balance_json_.contains("target_data") && balance_json_["target_data"].contains(idx_str)) {
                    target_data_[i] = balance_json_["target_data"][idx_str].get<double>();
                }
            }
            RCLCPP_INFO(this->get_logger(), "Balance config loaded.");
        }
        
        if (shm_connected_ && pose_ptr_) {
            for (int i = 0; i < 12; ++i) pose_ptr_[i] = target_data_[i];
        }
    }

    void save_balance_config() {
        std::ofstream out(balance_config_file_);
        out << balance_json_.dump(4);
        RCLCPP_INFO(this->get_logger(), "Balance config saved.");
    }

    // --- Shared Memory Loop ---
    void shm_loop() {
        while (running_ && rclcpp::ok()) {
            if (!shm_connected_) {
                if (access("/dev/shm/target_pose_shm", F_OK) != -1 && access("/dev/shm/eef_pos_shm", F_OK) != -1) {
                    fd_pose_ = shm_open("target_pose_shm", O_RDWR, 0666);
                    pose_ptr_ = (double*)mmap(0, 12 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_pose_, 0);
                    
                    fd_eef_ = shm_open("eef_pos_shm", O_RDONLY, 0666);
                    eef_ptr_ = (double*)mmap(0, 9 * sizeof(double), PROT_READ, MAP_SHARED, fd_eef_, 0);

                    if (pose_ptr_ != MAP_FAILED && eef_ptr_ != MAP_FAILED) {
                        for (int i = 0; i < 12; ++i) pose_ptr_[i] = target_data_[i]; 
                        shm_connected_ = true;
                        RCLCPP_INFO(this->get_logger(), "SHM Connected & Initial target_pose injected.");
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    // --- Camera Loop (tactile_ball_tracker_debug의 rpicam 파이프 방식 완벽 이식) ---
    void camera_loop(int idx) {
        SensorData& sensor = sensors_[idx];
        std::string source = sensor.source;

        // [방식 A] rpicam (CSI 라즈베리파이 카메라)
        if (source == "0" || source == "1") {
            std::string cmd = sensor.rpicam_cmd;
            if (cmd.empty()) {
                cmd = "rpicam-vid -t 0 --camera " + source + 
                      " --width 640 --height 400 --framerate 120" + 
                      " --codec yuv420 --denoise off --nopreview --awb daylight --shutter 20000 --gain 2.0 -o - 2>/dev/null";
            }

            while (running_ && rclcpp::ok()) {
                FILE* pipe = popen(cmd.c_str(), "r");
                if (!pipe) {
                    sensor.is_running = false;
                    RCLCPP_ERROR(this->get_logger(), "[Cam %d] Failed to open rpicam pipe.", idx);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    continue;
                }

                int width = 640, height = 400;
                int y_size = width * height;
                int frame_size = y_size * 3 / 2; // YUV420의 총 프레임 크기
                
                std::vector<uint8_t> buffer(frame_size);
                cv::Mat frame(height, width, CV_8UC1); // Grayscale (Y 채널만 사용할 용도)

                sensor.is_running = true;
                RCLCPP_INFO(this->get_logger(), "[Cam %d] RPi Cam started via popen successfully.", idx);

                while (running_ && rclcpp::ok()) {
                    // stdout 파이프에서 Raw 스트림을 정해진 프레임 사이즈만큼 읽어옵니다.
                    size_t bytes_read = fread(buffer.data(), 1, frame_size, pipe);
                    if (bytes_read != (size_t)frame_size) {
                        sensor.is_running = false;
                        break; // 스트림이 끊기면 다시 popen 하도록 while문을 빠져나감
                    }

                    // YUV420 포맷에서 가장 앞부분 Y_SIZE 만큼이 Grayscale 이미지입니다.
                    std::memcpy(frame.data, buffer.data(), y_size);

                    // 밝기 계산 (디스플레이 용)
                    cv::Scalar mean_scalar = cv::mean(frame);
                    sensor.current_brightness = mean_scalar[0];

                    if (is_paused_) continue;

                    // Adaptive threshold (same as tactile_ball_tracker)
                    {
                        cv::Rect roi = sensor.ref_roi;
                        if (roi.x + roi.width > frame.cols) roi.width = frame.cols - roi.x;
                        if (roi.y + roi.height > frame.rows) roi.height = frame.rows - roi.y;
                        cv::Mat ref_patch = frame(roi);
                        double raw_ref = cv::mean(ref_patch)[0];
                        sensor.ref_brightness = 0.995 * sensor.ref_brightness + 0.005 * raw_ref;
                        if (sensor.ref_baseline < 0.0) {
                            if (++sensor.warmup_count >= sensor.WARMUP_FRAMES)
                                sensor.ref_baseline = sensor.ref_brightness;
                        }
                        if (sensor.ref_baseline > 0.0) {
                            double ratio = sensor.ref_brightness / sensor.ref_baseline;
                            sensor.thresh = (int)(sensor.base_thresh * ratio);
                            if (sensor.thresh < 80)  sensor.thresh = 80;
                            if (sensor.thresh > 180) sensor.thresh = 180;
                        }
                    }

                    cv::Mat masked, thresh_img;
                    cv::bitwise_and(frame, sensor.mask, masked);
                    cv::threshold(masked, thresh_img, sensor.thresh, 255, cv::THRESH_BINARY);
                    
                    {
                        std::lock_guard<std::mutex> lock(data_mutex_);
                        sensor.current_area = cv::countNonZero(thresh_img);
                    }
                }
                
                pclose(pipe);
                sensor.is_running = false;
            }
        } 
        // [방식 B] 일반 USB 카메라 (V4L2)
        else {
            std::string dev_name = source.find("/dev/") == 0 ? source : "/dev/video" + source;

            auto apply_camera_settings = [&]() {
                std::string suffix = " > /dev/null 2>&1";
                system(("v4l2-ctl -d " + dev_name + " -c auto_exposure=1" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c exposure_time_absolute=150" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c gain=20" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c white_balance_automatic=0" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c backlight_compensation=0" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c power_line_frequency=0" + suffix).c_str());
            };

            cv::Mat frame, gray, masked, thresh_img;

            while (running_ && rclcpp::ok()) {
                if (!sensor.cap.isOpened()) {
                    sensor.is_running = false;
                    sensor.current_brightness = 0.0;
                    
                    sensor.cap.open(dev_name, cv::CAP_V4L2);
                    
                    if (sensor.cap.isOpened()) {
                        sensor.cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                        sensor.cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                        sensor.cap.set(cv::CAP_PROP_FRAME_HEIGHT, 400);
                        sensor.cap.set(cv::CAP_PROP_FPS, 120);
                        apply_camera_settings();
                        RCLCPP_INFO(this->get_logger(), "[Cam %d] USB Cam started via V4L2 successfully.", idx);
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        continue;
                    }
                }

                if (!sensor.cap.read(frame) || frame.empty()) {
                    sensor.is_running = false;
                    sensor.cap.release(); // 강제 릴리즈 후 재연결 유도
                    continue;
                }
                
                sensor.is_running = true;
                
                if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                else gray = frame.clone();

                cv::Scalar mean_scalar = cv::mean(gray);
                sensor.current_brightness = mean_scalar[0];

                if (is_paused_) continue;

                // Adaptive threshold (same as tactile_ball_tracker)
                {
                    cv::Rect roi = sensor.ref_roi;
                    if (roi.x + roi.width > gray.cols) roi.width = gray.cols - roi.x;
                    if (roi.y + roi.height > gray.rows) roi.height = gray.rows - roi.y;
                    cv::Mat ref_patch = gray(roi);
                    double raw_ref = cv::mean(ref_patch)[0];
                    sensor.ref_brightness = 0.995 * sensor.ref_brightness + 0.005 * raw_ref;
                    if (sensor.ref_baseline < 0.0) {
                        if (++sensor.warmup_count >= sensor.WARMUP_FRAMES)
                            sensor.ref_baseline = sensor.ref_brightness;
                    }
                    if (sensor.ref_baseline > 0.0) {
                        double ratio = sensor.ref_brightness / sensor.ref_baseline;
                        sensor.thresh = (int)(sensor.base_thresh * ratio);
                        if (sensor.thresh < 80)  sensor.thresh = 80;
                        if (sensor.thresh > 180) sensor.thresh = 180;
                    }
                }

                // [추가된 방어 코드] 들어온 이미지 크기가 마스크 크기(640x400)와 다르면 강제로 맞춥니다.
                if (gray.size() != sensor.mask.size()) {
                    cv::resize(gray, gray, sensor.mask.size());
                }

                cv::bitwise_and(gray, sensor.mask, masked);
                cv::threshold(masked, thresh_img, sensor.thresh, 255, cv::THRESH_BINARY);
                
                {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    sensor.current_area = cv::countNonZero(thresh_img);
                }
            }
        }
    }

    // --- 50 샘플 자동 수집 ---
    void record_measurement_auto(const std::string& label, double m_total, double m_add, double x, double y) {
        RCLCPP_INFO(this->get_logger(), "[%s] 자동 샘플링 50개 시작...", label.c_str());

        double area_sums[3] = {0.0, 0.0, 0.0};
        int sample_count = 0;
        int max_attempts = 500; // 카메라가 꺼져있을 때 무한루프 방지
        int attempts = 0;

        while (sample_count < 50 && attempts < max_attempts && rclcpp::ok()) {
            attempts++;
            
            bool all_cameras_ok = true;
            for (int i = 0; i < 3; ++i) {
                if (!sensors_[i].is_running) {
                    all_cameras_ok = false;
                }
            }

            if (all_cameras_ok) {
                std::lock_guard<std::mutex> lock(data_mutex_);
                for (int i = 0; i < 3; ++i) {
                    area_sums[i] += sensors_[i].current_area;
                }
                sample_count++;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }

        if (sample_count < 50) {
            RCLCPP_ERROR(this->get_logger(), "[%s] 샘플링 실패 (카메라 켜짐 상태를 확인하세요)", label.c_str());
            return;
        }

        std::lock_guard<std::mutex> lock(data_mutex_);
        MeasureData md;
        md.label = label;
        md.m_total = m_total;
        md.m_add = m_add;
        md.pos_x = x;
        md.pos_y = y;
        
        md.area[0] = area_sums[0] / 50.0;
        md.area[1] = area_sums[1] / 50.0;
        md.area[2] = area_sums[2] / 50.0;
        
        if (shm_connected_ && eef_ptr_) {
            md.eef_x[0] = eef_ptr_[0]; md.eef_y[0] = eef_ptr_[1];
            md.eef_x[1] = eef_ptr_[3]; md.eef_y[1] = eef_ptr_[4];
            md.eef_x[2] = eef_ptr_[6]; md.eef_y[2] = eef_ptr_[7];
        } else {
            RCLCPP_WARN(this->get_logger(), "EEF SHM not connected! (Using 0.0 for eef positions)");
            for(int i=0; i<3; ++i) { md.eef_x[i] = 0.0; md.eef_y[i] = 0.0; }
        }

        recorded_data_.push_back(md);
        RCLCPP_INFO(this->get_logger(), "==> [%s] 50샘플 평균 데이터 자동 기록 완료!", label.c_str());
    }

    // --- Force c0 측정 (판 무게 = 각 센서별 c0) ---
    void measure_force_c0() {
        if (sensors_[0].k_val == 0 || sensors_[1].k_val == 0 || sensors_[2].k_val == 0) {
            RCLCPP_ERROR(this->get_logger(), "k_slider not calibrated yet. Run Calculate first.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "[Force c0] Setting tilt to zero and collecting samples...");

        // 1. Set tilt to zero
        if (shm_connected_ && pose_ptr_) {
            target_data_[3] = 0.0; // roll = 0
            target_data_[4] = 0.0; // pitch = 0
            for (int i = 0; i < 12; ++i) pose_ptr_[i] = target_data_[i];
        }

        // 2. Wait for stabilization
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));

        // 3. Collect 200 area samples
        const int N_SAMPLES = 200;
        double area_sums[3] = {0.0, 0.0, 0.0};
        int collected = 0;
        int max_wait = 1000;

        while (collected < N_SAMPLES && max_wait-- > 0 && rclcpp::ok()) {
            bool all_ok = true;
            for (int i = 0; i < 3; ++i) {
                if (!sensors_[i].is_running) all_ok = false;
            }
            if (all_ok) {
                std::lock_guard<std::mutex> lock(data_mutex_);
                for (int i = 0; i < 3; ++i) {
                    area_sums[i] += sensors_[i].current_area;
                }
                collected++;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (collected < N_SAMPLES) {
            RCLCPP_ERROR(this->get_logger(), "[Force c0] Sampling failed (%d/%d)", collected, N_SAMPLES);
            return;
        }

        // 4. Compute c0 = k * (avg_area + b) per sensor
        for (int i = 0; i < 3; ++i) {
            double avg_area = area_sums[i] / N_SAMPLES;
            double k = sensors_[i].k_val;
            double b = sensors_[i].b_val;
            force_c0_[i] = k * (avg_area + b);
            // c1-c5 are zero for now
            for (int j = 1; j < 6; ++j) force_coeffs_[i][j] = 0.0;
            force_coeffs_[i][0] = force_c0_[i];
        }

        // 5. Save to JSON
        save_force_coeffs();

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "=== Force c0 측정 완료 (Zero Tilt, N=" << N_SAMPLES << ") ===\n\n";
        for (int i = 0; i < 3; ++i) {
            ss << "  Sensor " << i << " c0: " << force_c0_[i] << " gf\n";
        }
        ss << "\nSaved to force_coeffs.json (c1~c5 = 0)\n";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }

    void save_force_coeffs() {
        json j;
        j["sensors"] = json::array();
        for (int i = 0; i < 3; ++i) {
            json s;
            s["c0"] = force_coeffs_[i][0];
            s["c1"] = force_coeffs_[i][1];
            s["c2"] = force_coeffs_[i][2];
            s["c3"] = force_coeffs_[i][3];
            s["c4"] = force_coeffs_[i][4];
            s["c5"] = force_coeffs_[i][5];
            j["sensors"].push_back(s);
        }
        std::ofstream out("/home/kimdonghwi/capstone_ws_claude/force_coeffs.json");
        out << j.dump(4);
        RCLCPP_INFO(this->get_logger(), "force_coeffs.json saved.");
    }

    void clear_recorded_data() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        recorded_data_.clear();
        RCLCPP_INFO(this->get_logger(), "Recorded data cleared. (0 samples)");
    }

    std::string calculate_fit() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (recorded_data_.empty()) return "Error: No data recorded.";

        int N = recorded_data_.size();
        Eigen::MatrixXd A(3 * N, 8);
        Eigen::VectorXd Y(3 * N);

        for (int i = 0; i < N; ++i) {
            const auto& d = recorded_data_[i];
            
            A(3*i + 0, 0) = d.area[0];  A(3*i + 0, 1) = d.area[1];  A(3*i + 0, 2) = d.area[2];
            A(3*i + 0, 3) = 1.0;        A(3*i + 0, 4) = 1.0;        A(3*i + 0, 5) = 1.0;
            A(3*i + 0, 6) = 0.0;        A(3*i + 0, 7) = 0.0;
            Y(3*i + 0) = d.m_total;

            A(3*i + 1, 0) = d.eef_x[0]*d.area[0]; A(3*i + 1, 1) = d.eef_x[1]*d.area[1]; A(3*i + 1, 2) = d.eef_x[2]*d.area[2];
            A(3*i + 1, 3) = d.eef_x[0];           A(3*i + 1, 4) = d.eef_x[1];           A(3*i + 1, 5) = d.eef_x[2];
            A(3*i + 1, 6) = -d.m_total;           A(3*i + 1, 7) = 0.0;
            Y(3*i + 1) = d.m_add * d.pos_x;

            A(3*i + 2, 0) = d.eef_y[0]*d.area[0]; A(3*i + 2, 1) = d.eef_y[1]*d.area[1]; A(3*i + 2, 2) = d.eef_y[2]*d.area[2];
            A(3*i + 2, 3) = d.eef_y[0];           A(3*i + 2, 4) = d.eef_y[1];           A(3*i + 2, 5) = d.eef_y[2];
            A(3*i + 2, 6) = 0.0;                  A(3*i + 2, 7) = -d.m_total;
            Y(3*i + 2) = d.m_add * d.pos_y;
        }

        Eigen::VectorXd X = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(5);
        ss << "======= 캘리브레이션 결과 (Least Squares) =======\n\n";
        
        for (int i = 0; i < 3; ++i) {
            double Ki = X(i);
            double Bi = X(i + 3);
            double k_val = Ki;
            double b_val = (Ki != 0) ? (Bi / Ki) : 0.0;

            sensors_[i].k_val = k_val;
            sensors_[i].b_val = b_val;

            ss << "[Sensor " << i << "]\n";
            ss << "k_slider (gf/pixel) : " << k_val << "\n";
            ss << "b_slider (0gf시 밝기) : " << b_val << "\n\n";
        }

        ss << "[판 중심 오차 (Robot Origin 대비)]\n";
        ss << "Center X (cx) : " << X(6) << " m\n";
        ss << "Center Y (cy) : " << X(7) << " m\n";
        ss << "=================================================\n";
        
        return ss.str();
    }

    void set_paused(bool paused) { is_paused_ = paused; }
    std::vector<SensorData> sensors_;
    json balance_json_;
    double force_c0_[3] = {0.0, 0.0, 0.0};
    double force_coeffs_[3][6] = {{0}};
    QVector<QPushButton*> calib_btns_;

private:
    std::vector<std::thread> capture_threads_;
    std::thread shm_thread_;
    bool running_;
    bool is_paused_;
    bool shm_connected_;
    std::mutex data_mutex_;
    
    std::string tactile_config_file_;
    std::string balance_config_file_;
    double target_data_[12];
    
    int fd_pose_ = -1, fd_eef_ = -1;
    double* pose_ptr_ = nullptr;
    double* eef_ptr_ = nullptr;
    std::vector<MeasureData> recorded_data_;
};

// --- Qt5 UI Class ---
class BallEstUI : public QWidget {
public:
    BallEstUI(std::shared_ptr<BallCalibrationNode> node) : node_(node) {
        setWindowTitle("Ball Estimation & Calibration UI");
        QVBoxLayout* main_layout = new QVBoxLayout(this);

        // 1. Control & Config Box
        QGroupBox* ctrl_group = new QGroupBox("Global Controls & Config");
        QHBoxLayout* ctrl_layout = new QHBoxLayout();
        
        QPushButton* btn_resume = new QPushButton("▶ Start/Resume Vision");
        QPushButton* btn_pause = new QPushButton("⏸ Pause Vision");
        QPushButton* btn_save_bal = new QPushButton("Save Balance Config");
        QPushButton* btn_load_bal = new QPushButton("Load Balance Config");
        
        ctrl_layout->addWidget(btn_resume);
        ctrl_layout->addWidget(btn_pause);
        ctrl_layout->addWidget(btn_save_bal);
        ctrl_layout->addWidget(btn_load_bal);
        ctrl_group->setLayout(ctrl_layout);
        main_layout->addWidget(ctrl_group);

        // calibration buttons (비전 시작 전엔 비활성화)

        connect(btn_resume, &QPushButton::clicked, [this]() {
            node_->set_paused(false);
            for (auto* b : node_->calib_btns_) b->setEnabled(true);
        });
        connect(btn_pause, &QPushButton::clicked, [this]() {
            node_->set_paused(true);
            for (auto* b : node_->calib_btns_) b->setEnabled(false);
        });
        connect(btn_save_bal, &QPushButton::clicked, [this]() { node_->save_balance_config(); });
        connect(btn_load_bal, &QPushButton::clicked, [this]() { node_->load_balance_config("/home/kimdonghwi/capstone_ws_claude/balance_config.json"); });

        // 2. 상태 모니터링 텍스트 에디터 (실시간 상태 반영)
        status_text_ = new QTextEdit();
        status_text_->setReadOnly(true);
        status_text_->setMinimumHeight(150);
        status_text_->setStyleSheet("background-color: #2e2e2e; color: white;");
        main_layout->addWidget(status_text_);

        QTimer* status_timer = new QTimer(this);
        connect(status_timer, &QTimer::timeout, this, &BallEstUI::update_live_status);
        status_timer->start(100); // 10Hz 업데이트

        // 3. Calibration Steps Box
        QGroupBox* calib_group = new QGroupBox("Calibration Steps");
        QGridLayout* grid = new QGridLayout();
        
        QPushButton* btn_a = new QPushButton("(a) 공하중 측정 (0g)");
        QPushButton* btn_b = new QPushButton("(b) 판 합하중 측정 (234.7g)");
        node_->calib_btns_.append({btn_a, btn_b});
        btn_a->setEnabled(false);
        btn_b->setEnabled(false);
        QPushButton* btn_clear = new QPushButton("Clear All Data");
        btn_clear->setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;");
        grid->addWidget(btn_a, 0, 0, 1, 4);
        grid->addWidget(btn_clear, 0, 4, 1, 1);
        grid->addWidget(btn_b, 1, 0, 1, 5);

        connect(btn_clear, &QPushButton::clicked, [this]() {
            node_->clear_recorded_data();
            QMessageBox::information(this, "Cleared", "All recorded data has been cleared.");
        });

        // 스레드 기반 버튼 이벤트 헬퍼 람다
        auto connect_auto_btn = [this](QPushButton* btn, const std::string& label, double m_tot, double m_add, double x, double y) {
            connect(btn, &QPushButton::clicked, [this, btn, label, m_tot, m_add, x, y]() {
                std::thread([this, btn, label, m_tot, m_add, x, y]() {
                    QMetaObject::invokeMethod(btn, "setEnabled", Qt::QueuedConnection, Q_ARG(bool, false));
                    QMetaObject::invokeMethod(btn, "setText", Qt::QueuedConnection, Q_ARG(QString, "수집 중..."));
                    
                    node_->record_measurement_auto(label, m_tot, m_add, x, y);
                    
                    QMetaObject::invokeMethod(btn, "setText", Qt::QueuedConnection, Q_ARG(QString, "완료됨 (재수집 가능)"));
                    QMetaObject::invokeMethod(btn, "setEnabled", Qt::QueuedConnection, Q_ARG(bool, true));
                }).detach();
            });
        };

        connect_auto_btn(btn_a, "A(Zero)", 0.0, 0.0, 0, 0);
        connect_auto_btn(btn_b, "B(Plate)", 234.7, 0.0, 0, 0);

        // (c) 105.8g / (d) 140.4g Buttons
        std::vector<std::pair<double, double>> pos = {{0, 0.05}, {0.05, 0}, {0, -0.05}, {-0.05, 0}, {0, 0}};
        for (int i = 0; i < 5; ++i) {
            QString c_txt = QString("(c-%1) 105.8g [%2, %3]").arg(i+1).arg(pos[i].first).arg(pos[i].second);
            QString d_txt = QString("(d-%1) 140.4g [%2, %3]").arg(i+1).arg(pos[i].first).arg(pos[i].second);
            
            QPushButton* btn_c = new QPushButton(c_txt);
            QPushButton* btn_d = new QPushButton(d_txt);
            
            connect_auto_btn(btn_c, QString("C-%1").arg(i+1).toStdString(), 234.7 + 105.8, 105.8, pos[i].first, pos[i].second);
            connect_auto_btn(btn_d, QString("D-%1").arg(i+1).toStdString(), 234.7 + 140.4, 140.4, pos[i].first, pos[i].second);

            node_->calib_btns_.append({btn_c, btn_d});
            btn_c->setEnabled(false);
            btn_d->setEnabled(false);

            grid->addWidget(btn_c, 2, i);
            grid->addWidget(btn_d, 3, i);
        }

        calib_group->setLayout(grid);
        main_layout->addWidget(calib_group);

        // 4. Force c0 Measurement Box (NEW)
        QGroupBox* force_group = new QGroupBox("Force Sensor c0 (Plate Weight) Calibration");
        QHBoxLayout* force_layout = new QHBoxLayout();

        QPushButton* btn_c0_measure = new QPushButton("Measure Force c0 (Zero Tilt)");
        btn_c0_measure->setStyleSheet("background-color: #ff9800; padding: 12px; font-weight: bold; color: white;");
        btn_c0_measure->setEnabled(false);
        node_->calib_btns_.append(btn_c0_measure);
        force_layout->addWidget(btn_c0_measure);

        QLabel* lbl_c0_status = new QLabel("Requires: k,b computed (Calculate first), plate at zero tilt");
        lbl_c0_status->setStyleSheet("color: gray; font-size: 11px;");
        force_layout->addWidget(lbl_c0_status);
        force_group->setLayout(force_layout);
        main_layout->addWidget(force_group);

        connect(btn_c0_measure, &QPushButton::clicked, [this, lbl_c0_status]() {
            lbl_c0_status->setText("Measuring... hold still...");
            QApplication::processEvents();
            std::thread([this, lbl_c0_status]() {
                node_->measure_force_c0();
                QMetaObject::invokeMethod(lbl_c0_status, "setText", Qt::QueuedConnection,
                    Q_ARG(QString, QString("Done! c0 saved to force_coeffs.json")));
            }).detach();
        });

        // 5. Result & Execution Box
        QPushButton* btn_calc = new QPushButton("Calculate K, B, Offset (Least Squares)");
        btn_calc->setStyleSheet("background-color: lightblue; padding: 10px; font-weight: bold;");
        main_layout->addWidget(btn_calc);

        result_text_ = new QTextEdit();
        result_text_->setReadOnly(true);
        result_text_->setMinimumHeight(200);
        result_text_->setFontPointSize(12);
        main_layout->addWidget(result_text_);

        QPushButton* btn_apply = new QPushButton("Apply & Save to tactile_config.json");
        btn_apply->setStyleSheet("background-color: lightgreen; padding: 10px; font-weight: bold;");
        main_layout->addWidget(btn_apply);

        connect(btn_calc, &QPushButton::clicked, [this]() {
            std::string res = node_->calculate_fit();
            result_text_->setText(QString::fromStdString(res));
        });

        connect(btn_apply, &QPushButton::clicked, [this]() {
            node_->save_tactile_config();
            QMessageBox::information(this, "Saved", "tactile_config.json updated successfully.");
        });
    }

private:
    std::shared_ptr<BallCalibrationNode> node_;
    QTextEdit* status_text_;
    QTextEdit* result_text_;

    void update_live_status() {
        QString msg = "<b>[실시간 카메라 및 센서 상태]</b><br>";
        for (int i = 0; i < 3; ++i) {
            const auto& sensor = node_->sensors_[i];
            msg += QString("카메라 <b>%1</b>: ").arg(i);
            
            if (sensor.is_running) {
                msg += "<font color='lightgreen'>● 가동 중 (ON)</font>";
            } else {
                msg += "<font color='red'>X 중지됨 (OFF) - 자동 재연결 시도 중...</font>";
            }
            
            msg += QString(" | 밝기(Brightness): %1").arg(sensor.current_brightness, 5, 'f', 1);
            msg += QString(" | 측정 면적(Area): %1<br>").arg(sensor.current_area, 6, 'f', 1);
        }
        msg += "<br>※ 수집 버튼(A~D)을 누르면 카메라가 모두 켜져 있는 순간부터 자동으로 50프레임이 캡처됩니다.";
        
        status_text_->setHtml(msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BallCalibrationNode>();

    std::thread ros_thread([&node]() {
        rclcpp::spin(node);
    });

    QApplication app(argc, argv);
    BallEstUI ui(node);
    ui.resize(950, 750); 
    ui.show();

    int ret = app.exec();
    
    rclcpp::shutdown();
    if (ros_thread.joinable()) ros_thread.join();
    return ret;
}