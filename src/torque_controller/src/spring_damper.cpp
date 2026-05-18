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

// ==========================================
// 옵션: true일 경우 z 에러가 절대 위치가 아닌 
// 각 eef의 평균 z값 대비 상대 z값 차이로만 결정됨
#define Z_RELATIVE true 
// ==========================================

class SynchronizedSpringDamperNode : public rclcpp::Node {
public:
    SynchronizedSpringDamperNode() : Node("sync_spring_node"), running_(true) {
        init_parameters();
        init_shared_memory();
        init_pinocchio();

        triangle_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("finger_triangle_marker", 10);
        target_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("finger_target_markers", 10);

        // 시각화 타이머 (30Hz)
        vis_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), std::bind(&SynchronizedSpringDamperNode::publish_visualization_markers, this));

        // 고속 제어 루프 스레드 (~500Hz)
        control_thread_ = std::thread(&SynchronizedSpringDamperNode::shm_control_loop, this);
    }

    ~SynchronizedSpringDamperNode() {
        running_ = false;
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
        // Unmap Shared Memory
        munmap(state_ptr_, 2 * 12 * sizeof(double));
        munmap(cmd_ptr_, 12 * sizeof(double));
        if (pose_ptr_) munmap(pose_ptr_, 6 * sizeof(double)); // 차원수 6으로 변경
    }

private:
    std::thread control_thread_;
    std::atomic<bool> running_;
    std::mutex param_mutex_;

    // ROS Publishers
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr triangle_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr target_marker_pub_;
    rclcpp::TimerBase::SharedPtr vis_timer_;

    // Shared Memory Pointers
    double* state_ptr_ = nullptr;
    double* cmd_ptr_ = nullptr;
    double* pose_ptr_ = nullptr; // rp_ptr_ -> pose_ptr_
    bool pose_shared_memory_ = true;

    // Pinocchio 
    pinocchio::Model model_;
    pinocchio::Data data_;
    std::vector<pinocchio::FrameIndex> tip_ids_;

    // Control Parameters
    double K_ = 100.0, D_ = 3.0, K_rot_ = 0.0, gravity_comp_gain_ = 1.5;
    
    // 6D Target Variables (x, y, z, roll, pitch, yaw)
    double target_x_ = 0.0, target_y_ = 0.0, target_z_ = 0.252; // 초기 z값 설정
    double target_roll_ = 0.0, target_pitch_ = 0.0, target_yaw_ = 0.0;
    
    double TRI_RADIUS_ = 0.12;
    double F_FRIC_STATIC_ = 0.045, F_FRIC_BIAS_ = 0.0, FRIC_V_COMPENSATE_ = 10.0;
    
    Eigen::Matrix3d R_added_mat_;
    Eigen::Matrix3d target_R_ = Eigen::Matrix3d::Identity();
    Eigen::VectorXd D_joint_weight_;
    
    std::vector<Eigen::Vector3d> zero_target_tri_;
    std::vector<Eigen::Vector3d> curr_pos_;
    std::vector<Eigen::Vector3d> target_pos_actual_;

    void init_parameters() {
        D_joint_weight_ = Eigen::VectorXd(12);
        for(int i=0; i<3; ++i) {
            D_joint_weight_.segment<4>(i*4) << 2.0, 1.5, 1.0, 1.0;
        }

        // zero_target_bias_는 x, y, z를 받게 됨으로써 동적으로 계산되므로 제거됨
        zero_target_tri_ = {
            Eigen::Vector3d(-std::sqrt(3.0/4.0), 0.5, 0),
            Eigen::Vector3d(std::sqrt(3.0/4.0), 0.5, 0),
            Eigen::Vector3d(0, -1, 0)
        };
        curr_pos_.resize(3, Eigen::Vector3d::Zero());
        target_pos_actual_.resize(3, Eigen::Vector3d::Zero());
        
        update_rotation_matrix();
    }

    void init_shared_memory() {
        int fd_state = shm_open("dxl_state_shm", O_RDWR, 0666);
        int fd_cmd = shm_open("dxl_cmd_shm", O_RDWR, 0666);
        
        if (fd_state == -1 || fd_cmd == -1) {
            RCLCPP_ERROR(this->get_logger(), "State or Cmd Shared memory not found! Run interface first.");
            throw std::runtime_error("SHM Init Failed");
        }

        state_ptr_ = (double*)mmap(0, 2 * 12 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_state, 0);
        cmd_ptr_ = (double*)mmap(0, 12 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_cmd, 0);

        // Target Pose를 위한 Shared Memory (6차원으로 변경)
        int fd_pose = shm_open("target_pose_shm", O_RDWR, 0666);
        if (fd_pose == -1) {
            RCLCPP_WARN(this->get_logger(), "target_pose_shm not found. Creating a new one...");
            fd_pose = shm_open("target_pose_shm", O_CREAT | O_RDWR, 0666);
            if (ftruncate(fd_pose, 6 * sizeof(double)) == -1) {
                RCLCPP_ERROR(this->get_logger(), "Failed to allocate memory for target_pose_shm");
                pose_shared_memory_ = false;
            }
        }
        
        if (pose_shared_memory_) {
            pose_ptr_ = (double*)mmap(0, 6 * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_pose, 0);
            
            // 초기값 세팅 (쓰레기값 방지)
            pose_ptr_[0] = target_x_; pose_ptr_[1] = target_y_; pose_ptr_[2] = target_z_;
            pose_ptr_[3] = target_roll_; pose_ptr_[4] = target_pitch_; pose_ptr_[5] = target_yaw_;
            
            RCLCPP_INFO(this->get_logger(), "Successfully connected to 6D Target Pose shared memory.");
        }
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

    void update_rotation_matrix() {
        R_added_mat_ = Eigen::AngleAxisd(target_yaw_, Eigen::Vector3d::UnitZ())
                     * Eigen::AngleAxisd(target_pitch_, Eigen::Vector3d::UnitY())
                     * Eigen::AngleAxisd(target_roll_, Eigen::Vector3d::UnitX());
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
        pinocchio::Data::Matrix6x J(6, model_.nv);
        
        auto next_time = std::chrono::steady_clock::now();
        const auto loop_rate = std::chrono::microseconds(2000); // 500Hz (2ms)

        while (running_ && rclcpp::ok()) {
            // 1. Shared memory 업데이트 (x, y, z, roll, pitch, yaw)
            if (pose_shared_memory_ && pose_ptr_ != nullptr) {
                target_x_     = pose_ptr_[0];
                target_y_     = pose_ptr_[1];
                target_z_     = pose_ptr_[2];
                target_roll_  = pose_ptr_[3];
                target_pitch_ = pose_ptr_[4];
                target_yaw_   = pose_ptr_[5];
                update_rotation_matrix();
            }

            // 2. Pinocchio Forward Kinematics
            pinocchio::framesForwardKinematics(model_, data_, q);
            pinocchio::computeJointJacobians(model_, data_, q);
            const Eigen::VectorXd& tau_gravity = pinocchio::computeGeneralizedGravity(model_, data_, q);

            tau_task.setZero();
            tau_task_damper.setZero();

            // 3. 작업 공간 제어 준비 (목표 위치 및 현재 위치 계산)
            double mean_curr_z = 0.0;
            double mean_target_z = 0.0;
            Eigen::Vector3d target_center(target_x_, target_y_, target_z_);

            for (size_t i = 0; i < tip_ids_.size(); ++i) {
                auto tid = tip_ids_[i];
                curr_pos_[i] = data_.oMf[tid].translation();
                target_pos_actual_[i] = R_added_mat_ * zero_target_tri_[i] * TRI_RADIUS_ + target_center;

                #if Z_RELATIVE
                mean_curr_z += curr_pos_[i].z();
                mean_target_z += target_pos_actual_[i].z();
                #endif
            }

            #if Z_RELATIVE
            mean_curr_z /= 3.0;
            mean_target_z /= 3.0;
            #endif

            // 4. 에러 계산 및 자코비안 적용
            for (size_t i = 0; i < tip_ids_.size(); ++i) {
                auto tid = tip_ids_[i];
                Eigen::Matrix3d curr_R = data_.oMf[tid].rotation();
                
                Eigen::Vector3d error_p = target_pos_actual_[i] - curr_pos_[i];
                
                #if Z_RELATIVE
                // 절대 z 에러를 지우고, 평균대비 상대적인 z 에러만 사용
                error_p.z() = (target_pos_actual_[i].z() - mean_target_z) - (curr_pos_[i].z() - mean_curr_z);
                #endif

                Eigen::Matrix3d R_error_matrix = target_R_ * curr_R.transpose();
                Eigen::Vector3d error_R(
                    R_error_matrix(2, 1) - R_error_matrix(1, 2),
                    R_error_matrix(0, 2) - R_error_matrix(2, 0),
                    R_error_matrix(1, 0) - R_error_matrix(0, 1)
                );
                error_R *= 0.5;
                error_R.setZero();

                J.setZero();
                pinocchio::getFrameJacobian(model_, data_, tid, pinocchio::LOCAL_WORLD_ALIGNED, J);
                Eigen::MatrixXd J_v = J.topRows<3>();
                Eigen::MatrixXd J_w = J.bottomRows<3>();

                Eigen::Vector3d force_p = K_ * error_p;
                Eigen::Vector3d force_pd = D_ * (J_v * v);
                Eigen::Vector3d torque_R = K_rot_ * error_R;

                tau_task += J_v.transpose() * force_p + J_w.transpose() * torque_R;
                tau_task_damper += J_v.transpose() * force_pd;
            }

            // 5. 비선형 마찰 및 중력 보상 적용
            for (int i = 0; i < model_.nv; ++i) {
                double cosh_vel = std::cosh(FRIC_V_COMPENSATE_ * v[i]);
                double fric_comp = fric_compensation_function(tau_task[i], F_FRIC_STATIC_, F_FRIC_BIAS_) / cosh_vel;
                
                tau_total[i] = tau_task[i] + fric_comp 
                             + (gravity_comp_gain_ * tau_gravity[i]) 
                             - tau_task_damper[i] 
                             + (0.0 * D_joint_weight_[i] * v[i]); 
            }

            // 6. Shared Memory에 Cmd 인가
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