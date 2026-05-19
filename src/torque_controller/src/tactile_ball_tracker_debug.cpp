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
#include <atomic>
#include <QApplication>
#include <QWidget>
#include <QGridLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QTimer>
#include <QFont>
#include <signal.h>

using json = nlohmann::json;
using namespace std::chrono_literals;

struct SensorData {
    double force = 0.0;
    double last_area = 0.0;

    double k_val = 100.0;
    double b_val = 100.0;          // additive offset: force = k*A + b

    int target_fps = 60;
    std::atomic<bool> camera_ok{false};

    std::vector<cv::Point> trap_pts;
    cv::Mat mask;
    cv::UMat u_mask;

    int thresh = 120;
    int base_thresh = 120;

    // Adaptive threshold: ref_roi brightness tracking (drift compensation)
    cv::Rect ref_roi = cv::Rect(5, 5, 30, 30);
    double ref_brightness = 0.0;   // runtime EMA-filtered brightness in ref_roi
    double ref_baseline = -1.0;    // -1 = not initialized
    int warmup_count = 0;
    static constexpr int WARMUP_FRAMES = 200;

    double current_fps = 0.0;
    cv::Mat display_img;
};

// ─── Forward declare ───
class TrackerQtGui;

// ─── Main Node ───
class TactileBallTracker : public rclcpp::Node {
public:
    TactileBallTracker() : Node("tactile_ball_tracker") {
        last_time_ = this->now();

        load_config();
        init_shm();

        this->declare_parameter("fps_limit", 60);
        this->declare_parameter("filter_alpha", 1.0);
        this->declare_parameter("show_gui", false);
        this->declare_parameter("show_qt_gui", true);

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
        RCLCPP_INFO(this->get_logger(), "Tactile Ball Tracker Node Started.");
    }

    ~TactileBallTracker() {
        running_ = false;
        for (auto& th : capture_threads_) {
            if (th.joinable()) th.join();
        }

        // Save ref_baseline to JSON
        {
            double bl_vals[3];
            for (int i = 0; i < 3; ++i) bl_vals[i] = sensors_[i].ref_baseline;
            save_ref_baseline(bl_vals);
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

    SensorData* sensors() { return sensors_; }
    std::mutex& data_mutex() { return data_mutex_; }

    void set_k(int idx, double new_k) {
        double k_vals[3], b_vals[3];
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            sensors_[idx].k_val = new_k;
            for (int i = 0; i < 3; ++i) {
                k_vals[i] = sensors_[i].k_val;
                b_vals[i] = sensors_[i].b_val;
            }
        }
        save_config_json(k_vals, b_vals);
        RCLCPP_INFO(this->get_logger(), "Sensor %d: k = %.6f", idx+1, new_k);
    }

    void set_zero(int idx) {
        double k_vals[3], b_vals[3];
        double b_new;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            double a = sensors_[idx].last_area;
            sensors_[idx].b_val = -(sensors_[idx].k_val * a);
            b_new = sensors_[idx].b_val;
            for (int i = 0; i < 3; ++i) {
                k_vals[i] = sensors_[i].k_val;
                b_vals[i] = sensors_[i].b_val;
            }
            RCLCPP_INFO(this->get_logger(), "Sensor %d: zeroed, b = -k*A = %.2f (A=%.1f)", idx+1, b_new, a);
        }
        save_config_json(k_vals, b_vals);
    }

    void adjust_baseline(int idx, int delta) {
        double bl_vals[3];
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            sensors_[idx].ref_baseline += (double)delta;
            if (sensors_[idx].ref_baseline < 1.0) sensors_[idx].ref_baseline = 1.0;
            for (int i = 0; i < 3; ++i) bl_vals[i] = sensors_[i].ref_baseline;
            RCLCPP_INFO(this->get_logger(), "Sensor %d: ref_baseline += %d → %.1f", idx+1, delta, sensors_[idx].ref_baseline);
        }
        save_ref_baseline(bl_vals);
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
                sensors_[i].b_val = j["cameras"][i].value("b_slider", 0.0);

                // ref_roi for adaptive threshold
                if (j["cameras"][i].contains("ref_roi")) {
                    auto& rr = j["cameras"][i]["ref_roi"];
                    sensors_[i].ref_roi = cv::Rect(rr[0], rr[1], rr[2], rr[3]);
                }

                // restore ref_baseline (if saved, skip warmup)
                if (j["cameras"][i].contains("ref_baseline") && j["cameras"][i]["ref_baseline"].get<double>() > 0.0) {
                    sensors_[i].ref_baseline = j["cameras"][i]["ref_baseline"];
                    sensors_[i].warmup_count = SensorData::WARMUP_FRAMES;
                    RCLCPP_INFO(this->get_logger(), "[Sensor %d] Restored ref_baseline: %.2f", i, sensors_[i].ref_baseline);
                }

                sensors_[i].trap_pts.clear();
                for(auto& pt : j["cameras"][i]["trap_src"]) {
                    sensors_[i].trap_pts.push_back(cv::Point(pt[0], pt[1]));
                }
            }
            file.close();

            RCLCPP_INFO(this->get_logger(),
                "Config loaded. S1: k=%.4f b=%.1f | S2: k=%.4f b=%.1f | S3: k=%.4f b=%.1f",
                sensors_[0].k_val, sensors_[0].b_val,
                sensors_[1].k_val, sensors_[1].b_val,
                sensors_[2].k_val, sensors_[2].b_val);
        }
    }

    void save_config_json(const double k_vals[3], const double b_vals[3]) {
        std::ifstream file("/home/kimdonghwi/capstone_ws_claude/tactile_config.json");
        json j;
        if (file.is_open()) { file >> j; file.close(); }
        if (j.contains("cameras")) {
            for (int i = 0; i < 3; ++i) {
                if (j["cameras"].size() > (size_t)i) {
                    j["cameras"][i]["k_slider"] = k_vals[i];
                    j["cameras"][i]["b_slider"] = b_vals[i];
                }
            }
            std::ofstream out("/home/kimdonghwi/capstone_ws_claude/tactile_config.json");
            if (out.is_open()) { out << j.dump(4); out.close(); }
        }
    }

    void save_ref_baseline(const double bl_vals[3]) {
        std::ifstream file("/home/kimdonghwi/capstone_ws_claude/tactile_config.json");
        json j;
        if (file.is_open()) { file >> j; file.close(); }
        if (j.contains("cameras")) {
            for (int i = 0; i < 3; ++i) {
                if (j["cameras"].size() > (size_t)i) {
                    j["cameras"][i]["ref_baseline"] = bl_vals[i];
                }
            }
            std::ofstream out("/home/kimdonghwi/capstone_ws_claude/tactile_config.json");
            if (out.is_open()) { out << j.dump(4); out.close(); }
            RCLCPP_INFO(this->get_logger(), "ref_baseline saved to config.");
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
            printf("         TACTILE SENSOR TRACKER TUI                 \n");
            printf("====================================================\n");
            printf(" FPS Limit: %d\n\n", global_fps_limit);

            for(int i=0; i<3; ++i) {
                printf(" [Sensor %d] FPS:%5.1f | A:%7.1f | F:%7.2f gf | dF:%6.1f\n",
                       i+1, sensors_[i].current_fps, sensors_[i].last_area, sensors_[i].force,
                       shm_ptr_[3+i]);
                printf("            k=%.4f b=%.1f th=%d base_th=%d ref_bl=%.1f ref_br=%.1f warm=%d\n",
                       sensors_[i].k_val, sensors_[i].b_val,
                       sensors_[i].thresh, sensors_[i].base_thresh,
                       sensors_[i].ref_baseline, sensors_[i].ref_brightness,
                       sensors_[i].warmup_count);

                if(!sensors_[i].display_img.empty()) {
                    disp_imgs[i] = sensors_[i].display_img;  // ref-counted, no deep copy
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
        cv::Mat display_clone_;  // reused across frames (no per-frame allocation)

        auto process_frame = [&](cv::Mat& frame) {
            if (frame.empty()) return;

            cv::UMat u_frame, u_binary;

            try {
                frame.copyTo(u_frame);

                if (sensors_[idx].u_mask.empty() && !sensors_[idx].trap_pts.empty()) {
                    cv::Mat temp_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                    cv::fillConvexPoly(temp_mask, sensors_[idx].trap_pts, cv::Scalar(255));
                    temp_mask.copyTo(sensors_[idx].u_mask);
                    sensors_[idx].mask = temp_mask.clone();
                }

                // ── Adaptive threshold: ref_roi brightness tracking (τ ≈ 1s) ──
                cv::Rect roi = sensors_[idx].ref_roi;
                if (roi.x + roi.width > frame.cols) roi.width = frame.cols - roi.x;
                if (roi.y + roi.height > frame.rows) roi.height = frame.rows - roi.y;
                if (roi.width > 0 && roi.height > 0) {
                    cv::Mat ref_patch = frame(roi);
                    double raw_ref = cv::mean(ref_patch)[0];
                    // EMA: α ≈ 0.000278, τ ≈ 30s @120fps
                    sensors_[idx].ref_brightness = 0.999722 * sensors_[idx].ref_brightness + 0.000278 * raw_ref;

                    // warmup baseline
                    if (sensors_[idx].ref_baseline < 0.0) {
                        if (++sensors_[idx].warmup_count >= SensorData::WARMUP_FRAMES) {
                            sensors_[idx].ref_baseline = sensors_[idx].ref_brightness;
                            RCLCPP_INFO(this->get_logger(), "[Sensor %d] Warmup complete, ref_baseline=%.1f", idx, sensors_[idx].ref_baseline);
                        }
                    }
                    if (sensors_[idx].ref_baseline > 0.0) {
                        double ratio = sensors_[idx].ref_brightness / sensors_[idx].ref_baseline;
                        sensors_[idx].thresh = (int)(sensors_[idx].base_thresh * ratio);
                        if (sensors_[idx].thresh < 80)  sensors_[idx].thresh = 80;
                        if (sensors_[idx].thresh > 180) sensors_[idx].thresh = 180;
                    }
                }

                cv::threshold(u_frame, u_binary, sensors_[idx].thresh, 255, cv::THRESH_BINARY);

                if (!sensors_[idx].u_mask.empty()) {
                    cv::bitwise_and(u_binary, sensors_[idx].u_mask, u_binary);
                }

                u_binary.copyTo(display_clone_);

                double total_area = cv::countNonZero(display_clone_);
                double alpha = this->get_parameter("filter_alpha").as_double();

                {
                    std::lock_guard<std::mutex> lock(data_mutex_);

                    double smoothed_area = alpha * total_area + (1.0 - alpha) * sensors_[idx].last_area;
                    sensors_[idx].display_img = display_clone_;

                    double k = sensors_[idx].k_val;
                    double b = sensors_[idx].b_val;

                    sensors_[idx].last_area = smoothed_area;
                    sensors_[idx].force = k * smoothed_area + b;
                }

                u_frame.release();
                u_binary.release();

                sensors_[idx].camera_ok.store(true);

            } catch (const cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "[Sensor %d] OpenCV Exception: %s", idx + 1, e.what());
                u_frame.release();
                u_binary.release();
                sensors_[idx].u_mask.release();
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "[Sensor %d] Exception: %s", idx + 1, e.what());
            } catch (...) {
                RCLCPP_ERROR(this->get_logger(), "[Sensor %d] Unknown Exception.", idx + 1);
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

            cap.open(dev_name, cv::CAP_V4L2);
            if (cap.isOpened()) {
                cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                cap.set(cv::CAP_PROP_FPS, 120);

                cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
                cap.set(cv::CAP_PROP_EXPOSURE, 150);
                cap.set(cv::CAP_PROP_GAIN, 20);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to open USB camera (source: %s)", source.c_str());
                return;
            }

            auto apply_camera_settings = [dev_name]() {
                std::string suffix = " > /dev/null 2>&1";
                system(("v4l2-ctl -d " + dev_name + " -c auto_exposure=1" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c exposure_time_absolute=150" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c gain=20" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c white_balance_automatic=0" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c white_balance_temperature=4600" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c backlight_compensation=0" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c power_line_frequency=0" + suffix).c_str());
                system(("v4l2-ctl -d " + dev_name + " -c exposure_dynamic_framerate=0" + suffix).c_str());
            };

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

        double f1 = sensors_[0].force;
        double f2 = sensors_[1].force;
        double f3 = sensors_[2].force;

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

    friend class TrackerQtGui;
};

// ─── QT5 GUI ───
class TrackerQtGui : public QWidget {
    Q_OBJECT
public:
    TrackerQtGui(TactileBallTracker* node, QWidget* parent = nullptr)
        : QWidget(parent), node_(node)
    {
        setWindowTitle("Tactile Sensor Calibration");
        auto* main_layout = new QVBoxLayout(this);

        QFont bold_font;
        bold_font.setBold(true);

        // ── Live display (read-only labels) ──
        auto* live_group = new QGroupBox("Live Sensor Data");
        auto* live_layout = new QHBoxLayout();
        for (int col = 0; col < 3; ++col) {
            live_labels_[col] = new QLabel("S" + QString::number(col+1) + "\nA: ---- px\nF: ---- gf\n●");
            live_labels_[col]->setAlignment(Qt::AlignCenter);
            live_labels_[col]->setFont(bold_font);
            live_labels_[col]->setStyleSheet("color: red;");
            live_layout->addWidget(live_labels_[col]);
        }
        live_group->setLayout(live_layout);
        main_layout->addWidget(live_group);

        // ── 3x3 calibration grid (ALL writeable) ──
        auto* grid_group = new QGroupBox("Calibration Data (all editable)");
        auto* grid = new QGridLayout();
        QStringList headers = {"Sensor 1", "Sensor 2", "Sensor 3"};
        QStringList row_names = {"A [px]", "B [px]", "C [gf]"};

        for (int col = 0; col < 3; ++col) {
            auto* hdr = new QLabel(headers[col]);
            hdr->setFont(bold_font);
            hdr->setAlignment(Qt::AlignCenter);
            grid->addWidget(hdr, 0, col + 1);
        }
        for (int row = 0; row < 3; ++row) {
            auto* lbl = new QLabel(row_names[row]);
            lbl->setFont(bold_font);
            grid->addWidget(lbl, row + 1, 0);
        }

        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                auto* edit = new QLineEdit("0.0");
                edit->setAlignment(Qt::AlignRight);
                edit->setStyleSheet("background-color: #fffff0;");
                grid->addWidget(edit, row + 1, col + 1);
                cells_[row][col] = edit;
            }
        }
        grid_group->setLayout(grid);
        main_layout->addWidget(grid_group);

        // ── Per-sensor buttons (k calc + zero + baseline adjust) ──
        auto* btn_group = new QGroupBox("Per-Sensor Actions");
        auto* btn_layout = new QHBoxLayout();
        for (int col = 0; col < 3; ++col) {
            auto* vbox = new QVBoxLayout();
            auto* k_btn = new QPushButton("k 계산");
            auto* zero_btn = new QPushButton("영점");
            int sensor_idx = col;

            connect(k_btn, &QPushButton::clicked, [this, sensor_idx]() { calc_k(sensor_idx); });
            connect(zero_btn, &QPushButton::clicked, [this, sensor_idx]() { set_zero(sensor_idx); });

            auto* base_label = new QLabel("baseline");
            base_label->setAlignment(Qt::AlignCenter);

            auto* base_hbox = new QHBoxLayout();
            auto* bp10 = new QPushButton("+10");
            auto* bp1  = new QPushButton("+1");
            auto* bm1  = new QPushButton("-1");
            auto* bm10 = new QPushButton("-10");

            connect(bp10, &QPushButton::clicked, [this, sensor_idx]() { node_->adjust_baseline(sensor_idx, 10); });
            connect(bp1,  &QPushButton::clicked, [this, sensor_idx]() { node_->adjust_baseline(sensor_idx, 1); });
            connect(bm1,  &QPushButton::clicked, [this, sensor_idx]() { node_->adjust_baseline(sensor_idx, -1); });
            connect(bm10, &QPushButton::clicked, [this, sensor_idx]() { node_->adjust_baseline(sensor_idx, -10); });

            base_hbox->addWidget(bp10);
            base_hbox->addWidget(bp1);
            base_hbox->addWidget(bm1);
            base_hbox->addWidget(bm10);

            vbox->addWidget(new QLabel("Sensor " + QString::number(col+1)));
            vbox->addWidget(k_btn);
            vbox->addWidget(zero_btn);
            vbox->addWidget(base_label);
            vbox->addLayout(base_hbox);
            btn_layout->addLayout(vbox);
        }
        btn_group->setLayout(btn_layout);
        main_layout->addWidget(btn_group);

        // ── Refresh timer (10 Hz) ──
        auto* refresh_timer = new QTimer(this);
        connect(refresh_timer, &QTimer::timeout, this, &TrackerQtGui::refresh_display);
        refresh_timer->start(100);

        setMinimumWidth(550);
    }

private slots:
    void refresh_display() {
        auto* s = node_->sensors();
        std::lock_guard<std::mutex> lock(node_->data_mutex());

        for (int col = 0; col < 3; ++col) {
            double a = s[col].last_area;
            double force_val = s[col].k_val * a + s[col].b_val;
            bool ok = s[col].camera_ok.load();

            live_labels_[col]->setText(
                "S" + QString::number(col+1) +
                "\nA: " + QString::number(a, 'f', 1) + " px" +
                "\nF: " + QString::number(force_val, 'f', 2) + " gf" +
                "\nbl: " + QString::number(s[col].ref_baseline, 'f', 1) +
                "\n" + (ok ? "●" : "○"));
            live_labels_[col]->setStyleSheet(
                ok ? "color: green; font-weight: bold;" : "color: red; font-weight: bold;");
        }
    }

    void calc_k(int idx) {
        double a = cells_[0][idx]->text().toDouble();
        double b = cells_[1][idx]->text().toDouble();
        double c = cells_[2][idx]->text().toDouble();
        double denom = b - a;
        if (std::abs(denom) > 1.0) {
            double new_k = c / denom;
            node_->set_k(idx, new_k);
        }
    }

    void set_zero(int idx) {
        node_->set_zero(idx);
    }

private:
    TactileBallTracker* node_;
    QLineEdit* cells_[3][3];   // [row][col]: 0=A, 1=B, 2=C  (all writeable)
    QLabel* live_labels_[3];   // live display labels
};

// ─── Qt main ───
static TrackerQtGui* g_gui = nullptr;

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    // Qt application (must be created before node for signal handling)
    QApplication app(argc, argv);

    auto node = std::make_shared<TactileBallTracker>();

    bool show_qt = node->get_parameter("show_qt_gui").as_bool();
    if (show_qt) {
        g_gui = new TrackerQtGui(node.get());
        g_gui->show();
    }

    // ROS spin in separate thread, Qt event loop in main
    std::thread spin_thread([&node]() {
        rclcpp::spin(node);
    });

    int ret = app.exec();

    rclcpp::shutdown();
    if (spin_thread.joinable()) spin_thread.join();

    return ret;
}

#include "tactile_ball_tracker_debug.moc"
