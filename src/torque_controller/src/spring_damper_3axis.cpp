#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <Eigen/Dense>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <cmath>
#include <vector>

class SynchronizedSpringDamperNode : public rclcpp::Node {
public:
    SynchronizedSpringDamperNode() : Node("sync_spring_node"), running_(true) {
        init_parameters();
        init_shared_memory();
        init_pinocchio();

        triangle_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("finger_triangle_marker", 10);
        target_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("finger_target_markers", 10);

        vis_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), std::bind(&SynchronizedSpringDamperNode::publish_visualization_markers, this));

        control_thread_ = std::thread(&SynchronizedSpringDamperNode::shm_control_loop, this);
    }

    ~SynchronizedSpringDamperNode() {
        running_ = false;
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
        
        // 메모리 해제 및 SHM 파일 완벽 삭제
        munmap(state_ptr_, 2 * 12 * sizeof(double));
        munmap(cmd_ptr_, 12 * sizeof(double));
        if (pose_ptr_) munmap(pose_ptr_, 12 * sizeof(double));
        if (eef_ptr_) munmap(eef_ptr_, 6 * sizeof(double));
        if (eef_dot_ptr_) munmap(eef_dot_ptr_, 3 * sizeof(double));
        if (eef_force_ptr_) munmap(eef_force_ptr_, 3 * sizeof(double)); // [신설] 메모리 해제
        
        shm_unlink("target_pose_shm");
        shm_unlink("eef_pos_shm");
        shm_unlink("eef_dot_shm");
        shm_unlink("eef_force_shm"); // [신설] 언링크
    }

private:
    std::thread control_thread_;
    std::atomic<bool> running_;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr triangle_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr target_marker_pub_;
    rclcpp::TimerBase::SharedPtr vis_timer_;

    double* state_ptr_ = nullptr;
    double* cmd_ptr_ = nullptr;
    double* pose_ptr_ = nullptr; 
    double* eef_ptr_ = nullptr; 
    double* eef_dot_ptr_ = nullptr;
    double* eef_force_ptr_ = nullptr; // [신설] EEF 힘 저장용 포인터 추가 (3차원 double)
    bool pose_shared_memory_ = true;

    pinocchio::Model model_;
    pinocchio::Data data_;
    std::vector<pinocchio::FrameIndex> tip_ids_;

    double xyz_des_[3] = {0.0, 0.0, 0.0};
    double roll_des_ = 0.0;
    double pitch_des_ = 0.0;
    
    double K_task_ = 0.0;
    double K_task_2_ = 0.0;
    double D_task_ = 0.0;
    double K_ori_ = 0.0;

    double gravity_comp_gain_ = 2.0;
    double F_FRIC_STATIC_ = 0.045, F_FRIC_BIAS_ = 0.0, FRIC_V_COMPENSATE_ = 10.0;
    
    Eigen::VectorXd D_joint_weight_;
    std::vector<Eigen::Vector3d> curr_pos_;
    std::vector<Eigen::Vector3d> curr_vel_;
    std::vector<Eigen::Vector3d> target_pos_actual_;

    Eigen::Vector3d i0;
    Eigen::Vector3d j0;

    void init_parameters() {
        D_joint_weight_ = Eigen::VectorXd(12);
        for(int i=0; i<3; ++i) {
            D_joint_weight_.segment<4>(i*4) << 2.0, 1.5, 1.0, 1.0;
        }

        curr_pos_.resize(3, Eigen::Vector3d::Zero());
        curr_vel_.resize(3, Eigen::Vector3d::Zero());
        target_pos_actual_.resize(3, Eigen::Vector3d::Zero());

        i0 << 1.0, 0.0, 0.0;
        j0 << 0.0, 1.0, 0.0;
    }

    void init_shared_memory() {
        int fd_state = -1;
        int fd_cmd = -1;

        RCLCPP_INFO(this->get_logger(), "Waiting for 'dxl_state_shm' and 'dxl_cmd_shm' to be initialized by dynamixel_interface...");

        // 공유 메모리가 열릴 때까지 무한 대기 (Ctrl+C 누르면 rclcpp::ok()가 false가 되어 빠져나옴)
        while (rclcpp::ok()) {
            fd_state = shm_open("dxl_state_shm", O_RDWR, 0666);
            if (fd_state != -1) {
                fd_cmd = shm_open("dxl_cmd_shm", O_RDWR, 0666);
                if (fd_cmd != -1) {
                    // 두 파일 모두 성공적으로 오픈됨
                    break;
                }
                // fd_state는 열렸으나 fd_cmd가 실패한 경우, 메모리 누수 방지를 위해 닫고 재시도
                close(fd_state);
                fd_state = -1;
            }
            
            // 200ms 대기 후 재시도 (CPU 과부하 방지)
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        // 만약 공유 메모리가 열리기 전에 사용자가 Ctrl+C를 눌러 종료된 경우 예외 처리
        if (!rclcpp::ok()) {
            RCLCPP_WARN(this->get_logger(), "SHM Initialization cancelled by user.");
            throw std::runtime_error("SHM Init Cancelled");
        }

        RCLCPP_INFO(this->get_logger(), "Successfully connected to dxl_state_shm and dxl_cmd_shm!");

        // 기존 매핑 로직 유지
        state_ptr_ = (double*)mmap(0, 2 * 12 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_state, 0);
        cmd_ptr_ = (double*)mmap(0, 12 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_cmd, 0);

        // --- 1. Target Pose SHM (B->S, 11차원 강제 생성) ---
        shm_unlink("target_pose_shm"); 
        int fd_pose = shm_open("target_pose_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(fd_pose, 12 * sizeof(double));
        pose_ptr_ = (double*)mmap(0, 12 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_pose, 0);
        
        for (int i = 0; i < 8; ++i) pose_ptr_[i] = 0.0;
        pose_ptr_[8] = F_FRIC_STATIC_;
        pose_ptr_[9] = F_FRIC_BIAS_;
        pose_ptr_[10] = FRIC_V_COMPENSATE_;

        // --- 2. EEF Position SHM (S->B, 6차원 강제 생성) ---
        shm_unlink("eef_pos_shm");
        int fd_eef = shm_open("eef_pos_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(fd_eef, 6 * sizeof(double));
        eef_ptr_ = (double*)mmap(0, 6 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_eef, 0);

        // --- 3. EEF Dot SHM (S->B, 3차원 강제 생성) ---
        shm_unlink("eef_dot_shm");
        int fd_eef_dot = shm_open("eef_dot_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(fd_eef_dot, 3 * sizeof(double));
        eef_dot_ptr_ = (double*)mmap(0, 3 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_eef_dot, 0);
        for (int i = 0; i < 3; ++i) eef_dot_ptr_[i] = 0.0;

        // --- 4. EEF Force SHM [신설] (S->B, 3차원 강제 생성) ---
        shm_unlink("eef_force_shm");
        int fd_eef_force = shm_open("eef_force_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(fd_eef_force, 3 * sizeof(double));
        eef_force_ptr_ = (double*)mmap(0, 3 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_eef_force, 0);
        for (int i = 0; i < 3; ++i) eef_force_ptr_[i] = 0.0;

        RCLCPP_INFO(this->get_logger(), "S Node strictly created SHMs (target_pose_shm, eef_pos_shm, eef_dot_shm, eef_force_shm)");
    }

    void init_pinocchio() {
        std::string pkg_path = ament_index_cpp::get_package_share_directory("torque_controller");
        std::string urdf_path = pkg_path + "/urdf/hand_0926.urdf";
        
        pinocchio::urdf::buildModel(urdf_path, model_);
        model_.gravity.linear() = Eigen::Vector3d(0, 0, -9.81);
        data_ = pinocchio::Data(model_);

        std::vector<std::string> tip_names = {"FL1EEF", "FL2EEF", "FL3EEF"};
        for (const auto& name : tip_names) {
            if (model_.existFrame(name)) {
                tip_ids_.push_back(model_.getFrameId(name));
            } else {
                RCLCPP_ERROR(this->get_logger(), "Frame %s not found in URDF", name.c_str());
            }
        }
    }

    double fric_compensation_function(double x, double a, double b) {
        double abs_x = std::abs(x);
        double sign = (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0);
        
        if (abs_x >= 0 && abs_x < a) {
            return sign * ((b / a) * abs_x);
        } else if (abs_x >= a && abs_x < a + b) {
            return sign * (b - (abs_x - a));
        }
        return 0.0;
    }

    void shm_control_loop() {
        Eigen::Map<Eigen::VectorXd> q(state_ptr_, 12);
        Eigen::Map<Eigen::VectorXd> v(state_ptr_ + 12, 12);
        Eigen::Map<Eigen::VectorXd> tau_cmd(cmd_ptr_, 12);

        Eigen::VectorXd tau_task(model_.nv);
        Eigen::VectorXd tau_task_damper(model_.nv);
        Eigen::VectorXd tau_total(model_.nv);
        
        auto next_time = std::chrono::steady_clock::now();
        const auto loop_rate = std::chrono::microseconds(2000); 

        while (running_ && rclcpp::ok()) {
            if (pose_shared_memory_ && pose_ptr_ != nullptr) {
                xyz_des_[0]        = pose_ptr_[0];
                xyz_des_[1]        = pose_ptr_[1];
                xyz_des_[2]        = pose_ptr_[2];
                roll_des_          = pose_ptr_[3];
                pitch_des_         = pose_ptr_[4];
                K_task_            = pose_ptr_[5];
                D_task_            = pose_ptr_[6];
                K_ori_             = pose_ptr_[7];
                F_FRIC_STATIC_     = pose_ptr_[8];
                F_FRIC_BIAS_       = pose_ptr_[9];
                FRIC_V_COMPENSATE_ = pose_ptr_[10];
                K_task_2_          = pose_ptr_[11];
            }

            pinocchio::framesForwardKinematics(model_, data_, q);
            pinocchio::computeJointJacobians(model_, data_, q);
            const Eigen::VectorXd& tau_gravity = pinocchio::computeGeneralizedGravity(model_, data_, q);

            tau_task.setZero();
            tau_task_damper.setZero();

            Eigen::Matrix3d Rx = Eigen::AngleAxisd(roll_des_, Eigen::Vector3d::UnitX()).toRotationMatrix();
            Eigen::Matrix3d Ry = Eigen::AngleAxisd(pitch_des_, Eigen::Vector3d::UnitY()).toRotationMatrix();
            std::vector<Eigen::MatrixXd> J_v(3);
            std::vector<Eigen::MatrixXd> J_w(3);
            
            Eigen::Vector3d pos_center = Eigen::Vector3d::Zero();
            Eigen::Vector3d vel_center = Eigen::Vector3d::Zero();
            
            for (size_t i = 0; i < tip_ids_.size(); ++i) {
                auto tid = tip_ids_[i];
                pinocchio::Data::Matrix6x J(6, model_.nv);
                J.setZero();
                pinocchio::getFrameJacobian(model_, data_, tid, pinocchio::LOCAL_WORLD_ALIGNED, J);
                J_v[i] = J.topRows<3>();
                J_w[i] = J.bottomRows<3>();

                curr_pos_[i] = data_.oMf[tid].translation();
                curr_vel_[i] = J_v[i] * v;
                pos_center += curr_pos_[i] / 3;
                vel_center += curr_vel_[i] / 3;
            }

            for (size_t i = 0; i < tip_ids_.size(); ++i) {
                auto tid = tip_ids_[i];
                
                Eigen::Matrix3d curr_R = data_.oMf[tid].rotation();
                Eigen::Vector3d eef_y = curr_R.col(1); 

                Eigen::Vector3d i_pitch = Ry * i0;
                Eigen::Vector3d j_roll = Rx * j0;
                Eigen::Vector3d k_rp = i_pitch.cross(j_roll);
                Eigen::Vector3d e_rp = k_rp.cross(eef_y);

                // --- [SHM 기록] S -> B ---
                if (eef_ptr_ != nullptr) {
                    eef_ptr_[i * 2] = i_pitch.dot(curr_pos_[i]);
                    eef_ptr_[i * 2 + 1] = j_roll.dot(curr_pos_[i]);
                }

                // --- [신규 SHM 기록] S -> B (eef_dot_shm) ---
                if (eef_dot_ptr_ != nullptr) {
                    eef_dot_ptr_[i] = k_rp.dot(eef_y);
                }

                Eigen::Vector3d evx = i_pitch.dot(curr_vel_[i] - vel_center) * i_pitch;
                Eigen::Vector3d evy = j_roll.dot(curr_vel_[i] - vel_center) * j_roll;
                Eigen::Vector3d ez = k_rp.dot(pos_center - curr_pos_[i]) * k_rp;

                Eigen::Vector3d force_p = (K_task_ * ez) + (K_task_2_ * Eigen::Map<Eigen::Vector3d>(xyz_des_));
                Eigen::Vector3d force_pd = D_task_ * (evx + evy);           
                Eigen::Vector3d torque_R(K_ori_ * e_rp.x(), K_ori_ * e_rp.y(), 0.0); 

                tau_task += J_v[i].transpose() * force_p - J_w[i].transpose() * torque_R;
                tau_task_damper += J_v[i].transpose() * force_pd;
                target_pos_actual_[i] = Eigen::Vector3d(curr_pos_[i].x(), curr_pos_[i].y(), curr_pos_[i].z()) + Eigen::Map<Eigen::Vector3d>(xyz_des_);
            }

            for (int i = 0; i < model_.nv; ++i) {
                double cosh_vel = std::cosh(FRIC_V_COMPENSATE_ * v[i]);
                double fric_comp = fric_compensation_function(tau_task[i], F_FRIC_STATIC_, F_FRIC_BIAS_) / cosh_vel;
                
                tau_total[i] = tau_task[i] + fric_comp 
                             + (gravity_comp_gain_ * tau_gravity[i]) 
                             - tau_task_damper[i]; 
            }

            // --- [신설] 전류(토크)에 의한 말단 힘 분리 계산 및 eef_force_shm 기록 ---
            if (eef_force_ptr_ != nullptr) {
                for (size_t i = 0; i < tip_ids_.size(); ++i) {
                    auto tid = tip_ids_[i];
                    
                    // 각 손가락의 Jacobian 분리하여 연산공간 매트릭스 구성 M = J * J^T
                    Eigen::Matrix3d M = J_v[i] * J_v[i].transpose();
                    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(M);
                    
                    if (qr.isInvertible()) {
                        // F_i = (J_v * J_v^T)^-1 * J_v * tau (Pseudo-inverse 활용한 작업공간 힘 추정)
                        Eigen::Vector3d force_N = qr.solve(J_v[i] * tau_total);
                        
                        // 말단 프레임 y축 벡터 추출
                        Eigen::Matrix3d curr_R = data_.oMf[tid].rotation();
                        Eigen::Vector3d eef_y = curr_R.col(1);
                        
                        // y축 벡터에 투영 (Dot product)
                        double force_projected_N = force_N.dot(eef_y);
                        
                        // 뉴턴(N) 단위를 xh430 v350 r 물리적 환산 상수에 따라 그램중(gf) 단위로 변환
                        // 1 N = 1000 / 9.80665 gf ≈ 101.97162 gf
                        double force_gf = force_projected_N * 101.97162;
                        eef_force_ptr_[i] = force_gf;
                    } else {
                        // Jacobian 역행렬 계산 불가능(Singular 상태 등)할 경우 안전을 위해 힘을 0으로 처리
                        eef_force_ptr_[i] = 0.0;
                    }
                }
            }

            tau_cmd = tau_total;
            next_time += loop_rate;
            std::this_thread::sleep_until(next_time);
        }
    }

    void publish_visualization_markers() {
        auto now = this->now();

        visualization_msgs::msg::Marker tri_marker;
        tri_marker.header.frame_id = "base_link";
        tri_marker.header.stamp = now;
        tri_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        tri_marker.id = 99;
        tri_marker.scale.x = 0.003;
        tri_marker.color.g = 1.0; tri_marker.color.a = 1.0;

        for (const auto& p : curr_pos_) {
            geometry_msgs::msg::Point pt; pt.x = p.x(); pt.y = p.y(); pt.z = p.z();
            tri_marker.points.push_back(pt);
        }
        if (!curr_pos_.empty()) {
            geometry_msgs::msg::Point pt; pt.x = curr_pos_[0].x(); pt.y = curr_pos_[0].y(); pt.z = curr_pos_[0].z();
            tri_marker.points.push_back(pt);
        }
        triangle_marker_pub_->publish(tri_marker);

        visualization_msgs::msg::Marker target_marker;
        target_marker.header.frame_id = "base_link";
        target_marker.header.stamp = now;
        target_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        target_marker.id = 100;
        target_marker.scale.x = 0.015; target_marker.scale.y = 0.015; target_marker.scale.z = 0.015;
        target_marker.color.r = 1.0; target_marker.color.a = 0.8;

        for (const auto& p : target_pos_actual_) {
            geometry_msgs::msg::Point pt; pt.x = p.x(); pt.y = p.y(); pt.z = p.z();
            target_marker.points.push_back(pt);
        }
        target_marker_pub_->publish(target_marker);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SynchronizedSpringDamperNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}