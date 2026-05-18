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
};

// 측정 데이터를 저장할 구조체
struct MeasureData {
    std::string label;
    double mass_g;
    double pos_x;
    double pos_y;
    double area[3];
};

class BallCalibrationNode : public rclcpp::Node {
public:
    BallCalibrationNode() : Node("ball_calibration_node"), is_paused_(true), running_(true) {
        load_tactile_config("tactile_config.json");
        load_balance_config("balance_config.json");
        init_shm();

        // 비전 처리 스레드 시작
        for (int i = 0; i < 3; ++i) {
            capture_threads_.emplace_back(&BallCalibrationNode::camera_loop, this, i);
        }
    }

    ~BallCalibrationNode() {
        running_ = false;
        for (auto& th : capture_threads_) {
            if (th.joinable()) th.join();
        }
        if (pose_ptr_ && pose_ptr_ != MAP_FAILED) {
            munmap(pose_ptr_, 12 * sizeof(double));
        }
        if (fd_pose_ != -1) close(fd_pose_);
    }

    // --- JSON Load / Save ---
    void load_tactile_config(const std::string& filename) {
        tactile_config_file_ = filename;
        std::ifstream file(filename);
        if (!file.is_open()) return;
        json j; file >> j;
        
        sensors_.resize(3);
        for (int i = 0; i < 3 && i < j["cameras"].size(); ++i) {
            auto& cam = j["cameras"][i];
            sensors_[i].b_val = cam.value("b_slider", 0.0);
            sensors_[i].k_val = cam.value("k_slider", 1.0);
            sensors_[i].source = cam.value("source", "0");
            sensors_[i].rpicam_cmd = cam.value("rpicam_cmd", "");
            
            sensors_[i].trap_pts.clear();
            for (auto& pt : cam["trap_src"]) {
                sensors_[i].trap_pts.push_back(cv::Point(pt[0], pt[1]));
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
        }
        for (int i = 0; i < 3; ++i) {
            j["cameras"][i]["k_slider"] = sensors_[i].k_val;
            j["cameras"][i]["b_slider"] = sensors_[i].b_val;
        }
        std::ofstream out(tactile_config_file_);
        out << j.dump(4);
    }

    void load_balance_config(const std::string& filename) {
        balance_config_file_ = filename;
        std::ifstream file(filename);
        if (!file.is_open()) {
            RCLCPP_WARN(this->get_logger(), "balance_config.json not found. Using defaults.");
            return;
        }
        file >> balance_json_;
        RCLCPP_INFO(this->get_logger(), "Balance config loaded automatically at init.");
    }

    void save_balance_config() {
        std::ofstream out(balance_config_file_);
        out << balance_json_.dump(4);
        RCLCPP_INFO(this->get_logger(), "Balance config saved.");
    }

    // --- Shared Memory ---
    void init_shm() {
        fd_pose_ = shm_open("target_pose_shm", O_RDONLY, 0666);
        if (fd_pose_ != -1) {
            pose_ptr_ = (double*)mmap(0, 12 * sizeof(double), PROT_READ, MAP_SHARED, fd_pose_, 0);
        }
    }

    // --- Camera Loop ---
    void camera_loop(int idx) {
        SensorData& sensor = sensors_[idx];
        std::string dev_name = sensor.source.find("/dev/") == 0 ? sensor.source : "/dev/video" + sensor.source;

        sensor.cap.open(dev_name, cv::CAP_V4L2);
        if (!sensor.cap.isOpened()) return;

        sensor.cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        sensor.cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        sensor.cap.set(cv::CAP_PROP_FRAME_HEIGHT, 400);
        sensor.cap.set(cv::CAP_PROP_FPS, 120);

        // 하드웨어 파라미터 강제
        auto apply_camera_settings = [&]() {
            std::string suffix = " > /dev/null 2>&1";
            system(("v4l2-ctl -d " + dev_name + " -c auto_exposure=1" + suffix).c_str());
            system(("v4l2-ctl -d " + dev_name + " -c exposure_time_absolute=150" + suffix).c_str());
            system(("v4l2-ctl -d " + dev_name + " -c gain=20" + suffix).c_str());
            system(("v4l2-ctl -d " + dev_name + " -c white_balance_automatic=0" + suffix).c_str());
            system(("v4l2-ctl -d " + dev_name + " -c backlight_compensation=0" + suffix).c_str());
            system(("v4l2-ctl -d " + dev_name + " -c power_line_frequency=0" + suffix).c_str());
        };
        apply_camera_settings();

        cv::Mat frame, gray, masked;
        auto last_enforce_time = std::chrono::steady_clock::now();

        while (running_ && rclcpp::ok()) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_enforce_time).count() >= 10.0) {
                apply_camera_settings();
                last_enforce_time = now;
            }

            if (!sensor.cap.read(frame)) continue;
            
            if (is_paused_) continue; // Paused 상태면 이미지 처리 안 함

            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::bitwise_and(gray, sensor.mask, masked);
            
            cv::Mat thresh_img;
            cv::threshold(masked, thresh_img, 120, 255, cv::THRESH_BINARY);
            
            std::lock_guard<std::mutex> lock(data_mutex_);
            sensor.current_area = cv::countNonZero(thresh_img);
        }
    }

    // --- Measurement & Fit Logic ---
    void record_measurement(const std::string& label, double mass_g, double x, double y) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        MeasureData md;
        md.label = label;
        md.mass_g = mass_g;
        md.pos_x = x;
        md.pos_y = y;
        md.area[0] = sensors_[0].current_area;
        md.area[1] = sensors_[1].current_area;
        md.area[2] = sensors_[2].current_area;
        recorded_data_.push_back(md);
    }

    std::string calculate_fit() {
        if (!pose_ptr_ || pose_ptr_ == MAP_FAILED) return "Error: SHM not connected.";
        if (recorded_data_.empty()) return "Error: No data recorded.";

        // EEF 위치 (예시 인덱싱, 실제 모델에 맞게 수정 필요)
        Eigen::Vector2d p1(pose_ptr_[0], pose_ptr_[1]);
        Eigen::Vector2d p2(pose_ptr_[4], pose_ptr_[5]);
        Eigen::Vector2d p3(pose_ptr_[8], pose_ptr_[9]);

        // Least Squares로 K, B 추정 (Model: gf = k * (area + b))
        // 식 전개: gf/k - b = area  => area = (1/k)*gf - b
        // M matrix * [1/k, -b]^T = Area Vector
        std::stringstream ss;
        ss << "=== 캘리브레이션 결과 ===\n";

        for (int i = 0; i < 3; ++i) {
            Eigen::MatrixXd A(recorded_data_.size(), 2);
            Eigen::VectorXd Y(recorded_data_.size());

            for (size_t j = 0; j < recorded_data_.size(); ++j) {
                // 토크 밸런스를 이용해 각 손가락에 걸린 실제 Force(gf) 계산
                double total_mass = recorded_data_[j].mass_g;
                double tx = recorded_data_[j].pos_x;
                double ty = recorded_data_[j].pos_y;
                
                // 간략한 분배 식 (정확한 역학 모델은 3x3 행렬 역행렬 사용)
                Eigen::Matrix3d PosMat;
                PosMat << 1, 1, 1,
                          p1.x(), p2.x(), p3.x(),
                          p1.y(), p2.y(), p3.y();
                Eigen::Vector3d Target(total_mass, total_mass * tx, total_mass * ty);
                Eigen::Vector3d Forces = PosMat.inverse() * Target;
                
                double actual_f = Forces(i);
                
                A(j, 0) = actual_f; // X항: Force
                A(j, 1) = 1.0;      // 상수항
                Y(j) = recorded_data_[j].area[i]; // Y항: Area
            }

            // A * [1/k, -b]^T = Y
            Eigen::Vector2d x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
            double inv_k = x(0);
            double minus_b = x(1);

            double k_val = 1.0 / inv_k;
            double b_val = -minus_b;

            sensors_[i].k_val = k_val;
            sensors_[i].b_val = b_val;

            ss << "Sensor " << i << " | K: " << k_val << " | B: " << b_val << "\n";
        }
        
        return ss.str();
    }

    void set_paused(bool paused) { is_paused_ = paused; }
    
    // Public Members for UI Access
    std::vector<SensorData> sensors_;
    json balance_json_;
    
private:
    std::vector<std::thread> capture_threads_;
    bool running_;
    bool is_paused_;
    std::mutex data_mutex_;
    std::string tactile_config_file_;
    std::string balance_config_file_;
    
    int fd_pose_ = -1;
    double* pose_ptr_ = nullptr;
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

        connect(btn_resume, &QPushButton::clicked, [this]() { node_->set_paused(false); });
        connect(btn_pause, &QPushButton::clicked, [this]() { node_->set_paused(true); });
        connect(btn_save_bal, &QPushButton::clicked, [this]() { node_->save_balance_config(); });
        connect(btn_load_bal, &QPushButton::clicked, [this]() { node_->load_balance_config("balance_config.json"); });

        // 2. Calibration Steps Box
        QGroupBox* calib_group = new QGroupBox("Calibration Steps");
        QGridLayout* grid = new QGridLayout();
        
        QPushButton* btn_a = new QPushButton("(a) 공하중 측정 (0g)");
        QPushButton* btn_b = new QPushButton("(b) 판 합하중 측정 (234.7g)");
        grid->addWidget(btn_a, 0, 0, 1, 5);
        grid->addWidget(btn_b, 1, 0, 1, 5);

        // (c) 105.8g / (d) 140.4g Buttons
        std::vector<std::pair<double, double>> pos = {{0, 0.1}, {0.1, 0}, {0, -0.1}, {-0.1, 0}, {0, 0}};
        for (int i = 0; i < 5; ++i) {
            QString c_txt = QString("(c-%1) 105.8g [%2, %3]").arg(i+1).arg(pos[i].first).arg(pos[i].second);
            QString d_txt = QString("(d-%1) 140.4g [%2, %3]").arg(i+1).arg(pos[i].first).arg(pos[i].second);
            
            QPushButton* btn_c = new QPushButton(c_txt);
            QPushButton* btn_d = new QPushButton(d_txt);
            
            connect(btn_c, &QPushButton::clicked, [this, i, pos]() {
                node_->record_measurement(QString("C-%1").arg(i+1).toStdString(), 234.7 + 105.8, pos[i].first, pos[i].second);
            });
            connect(btn_d, &QPushButton::clicked, [this, i, pos]() {
                node_->record_measurement(QString("D-%1").arg(i+1).toStdString(), 234.7 + 140.4, pos[i].first, pos[i].second);
            });
            
            grid->addWidget(btn_c, 2, i);
            grid->addWidget(btn_d, 3, i);
        }
        
        connect(btn_a, &QPushButton::clicked, [this]() { node_->record_measurement("A(Zero)", 0.0, 0, 0); });
        connect(btn_b, &QPushButton::clicked, [this]() { node_->record_measurement("B(Plate)", 234.7, 0, 0); });

        calib_group->setLayout(grid);
        main_layout->addWidget(calib_group);

        // 3. Result & Execution Box
        QPushButton* btn_calc = new QPushButton("Calculate K, B (Least Squares)");
        btn_calc->setStyleSheet("background-color: lightblue; padding: 10px; font-weight: bold;");
        main_layout->addWidget(btn_calc);

        result_text_ = new QTextEdit();
        result_text_->setReadOnly(true);
        result_text_->setMinimumHeight(150); // 복사하기 편하게 큰 텍스트창 설정
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
            QMessageBox::information(this, "Saved", "tactile_config.json updated with new K and B values.");
        });
    }

private:
    std::shared_ptr<BallCalibrationNode> node_;
    QTextEdit* result_text_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BallCalibrationNode>();

    std::thread ros_thread([&node]() {
        rclcpp::spin(node);
    });

    QApplication app(argc, argv);
    BallEstUI ui(node);
    ui.resize(900, 600);
    ui.show();

    int ret = app.exec();
    
    rclcpp::shutdown();
    if (ros_thread.joinable()) ros_thread.join();
    return ret;
}