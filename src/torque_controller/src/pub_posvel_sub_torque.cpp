#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cerrno>
#include <cstring>

// Shared Memory를 위한 헤더
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "dynamixel_sdk/dynamixel_sdk.h"

using namespace std::chrono_literals;

// --- Control Table (XH/XM Series) ---
#define ADDR_RETURN_DELAY_TIME      9
#define ADDR_OPERATING_MODE         11
#define ADDR_TORQUE_ENABLE          64
#define ADDR_GOAL_CURRENT           102
#define ADDR_GOAL_POSITION          116
#define ADDR_PRESENT_VELOCITY       128 
#define ADDR_PRESENT_POSITION       132

#define LEN_TORQUE_ENABLE           1
#define LEN_GOAL_CURRENT            2
#define LEN_GOAL_POSITION           4
#define LEN_POS_VEL_READ            8
#define LEN_PRESENT_VELOCITY        4
#define LEN_PRESENT_POSITION        4

#define TORQUE_ENABLE               1
#define TORQUE_DISABLE              0
#define OP_MODE_CURRENT             0
#define OP_MODE_POSITION            3

#define PROTOCOL_VERSION            2.0
#define BAUDRATE                    4000000
#define DEVICENAME                  "/dev/ttyUSB0"

#define KT_CONSTANT                 1.0
#define CURRENT_UNIT_A              0.00415123456
#define VELOCITY_UNIT_RPM           0.229
#define RPM_TO_RAD_SEC              0.104719755
#define DEG_TO_DXL                  (1.0 / 0.088)

class DynamixelSHMNode : public rclcpp::Node
{
public:
    DynamixelSHMNode() : Node("dynamixel_shm_node")
    {
        // 1. 파라미터 및 변수 초기화
        target_ids_ = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        // offsets_ = {180, 180.0, 180.0, 180.0, 112.5, 180.0, 180.0, 180.0, 247.5, 180.0, 180.0, 180.0};
        offsets_ = {337.5, 180.0, 180.0, 180.0, 112.5, 180.0, 180.0, 180.0, 247.5, 180.0, 180.0, 180.0};
        joint_names_ = {"F11", "F12", "F13", "F14", "F21", "F22", "F23", "F24", "F31", "F32", "F33", "F34"};

        // 2. Shared Memory 초기화 (Python과 호환되는 이름 사용)
        init_shared_memory();

        // 3. ROS 2 Publisher
        pub_joint_state_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);

        // 4. Dynamixel SDK 초기화
        portHandler_ = dynamixel::PortHandler::getPortHandler(DEVICENAME);
        packetHandler_ = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

        if (!portHandler_->openPort() || !portHandler_->setBaudRate(BAUDRATE)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open port or set baudrate!");
            return;
        }

        groupSyncRead_ = new dynamixel::GroupSyncRead(portHandler_, packetHandler_, ADDR_PRESENT_VELOCITY, LEN_POS_VEL_READ);
        groupSyncWriteCurrent_ = new dynamixel::GroupSyncWrite(portHandler_, packetHandler_, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT);
        groupSyncWritePosition_ = new dynamixel::GroupSyncWrite(portHandler_, packetHandler_, ADDR_GOAL_POSITION, LEN_GOAL_POSITION);
        groupSyncWriteTorque_ = new dynamixel::GroupSyncWrite(portHandler_, packetHandler_, ADDR_TORQUE_ENABLE, LEN_TORQUE_ENABLE);

        setup_dynamixels();

        // 5. 메인 루프 타이머 (1ms = 1000Hz)
        timer_ = this->create_wall_timer(1ms, std::bind(&DynamixelSHMNode::control_loop, this));
    }

    ~DynamixelSHMNode()
    {
        set_torque(false);
        close_shared_memory();
        portHandler_->closePort();
    }

private:
    void init_shared_memory()
    {
        // State SHM: 2행 12열 double (192 bytes)
        size_t state_size = 2 * 12 * sizeof(double);
        shm_state_fd_ = shm_open("dxl_state_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(shm_state_fd_, state_size);
        state_ptr_ = (double*)mmap(0, state_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_state_fd_, 0);
        memset(state_ptr_, 0, state_size);

        // Command SHM: 12열 double (96 bytes)
        size_t cmd_size = 12 * sizeof(double);
        shm_cmd_fd_ = shm_open("dxl_cmd_shm", O_CREAT | O_RDWR, 0666);
        ftruncate(shm_cmd_fd_, cmd_size);
        cmd_ptr_ = (double*)mmap(0, cmd_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_cmd_fd_, 0);
        memset(cmd_ptr_, 0, cmd_size);

        RCLCPP_INFO(this->get_logger(), "Shared Memory Initialized (dxl_state_shm, dxl_cmd_shm)");
    }

    void close_shared_memory()
    {
        munmap(state_ptr_, 2 * 12 * sizeof(double));
        munmap(cmd_ptr_, 12 * sizeof(double));
        shm_unlink("dxl_state_shm");
        shm_unlink("dxl_cmd_shm");
    }

    void setup_dynamixels()
    {
        for (uint8_t id : target_ids_) {
            packetHandler_->write1ByteTxRx(portHandler_, id, ADDR_RETURN_DELAY_TIME, 0);
            packetHandler_->write1ByteTxRx(portHandler_, id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE);

            // 1, 5, 9번 모터는 Position Mode, 나머지는 Current Mode
            uint8_t mode = (id == 1 || id == 5 || id == 9) ? OP_MODE_POSITION : OP_MODE_CURRENT;
            packetHandler_->write1ByteTxRx(portHandler_, id, ADDR_OPERATING_MODE, mode);
            groupSyncRead_->addParam(id);
        }
        set_torque(true);
        groupSyncWritePosition_->clearParam();
        // ID 1: -60도, ID 5: 0도, ID 9: 60도 (오프셋 적용)
        std::vector<std::pair<uint8_t, double>> targets = {{1, -50.0}, {5, 50}, {9, 0.0}};
        
        for (auto& target : targets) {
            uint8_t id = target.first;
            double target_deg = target.second;
            int32_t pos_raw = (int32_t)((target_deg + offsets_[id-1]) * DEG_TO_DXL);
            
            uint8_t param[4];
            param[0] = DXL_LOBYTE(DXL_LOWORD(pos_raw));
            param[1] = DXL_HIBYTE(DXL_LOWORD(pos_raw));
            param[2] = DXL_LOBYTE(DXL_HIWORD(pos_raw));
            param[3] = DXL_HIBYTE(DXL_HIWORD(pos_raw));
            groupSyncWritePosition_->addParam(id, param);
        }
        groupSyncWritePosition_->txPacket();
    }

    void set_torque(bool enable)
    {
        uint8_t data = enable ? TORQUE_ENABLE : TORQUE_DISABLE;
        groupSyncWriteTorque_->clearParam();
        for (uint8_t id : target_ids_) groupSyncWriteTorque_->addParam(id, &data);
        groupSyncWriteTorque_->txPacket();
    }

    void control_loop()
    {
        // 1. READ Dynamixel & Write to SHM (State)
        if (groupSyncRead_->txRxPacket() == COMM_SUCCESS) {
            auto msg = sensor_msgs::msg::JointState();
            msg.header.stamp = this->now();

            for (size_t i = 0; i < target_ids_.size(); ++i) {
                uint8_t id = target_ids_[i];
                if (groupSyncRead_->isAvailable(id, ADDR_PRESENT_VELOCITY, LEN_POS_VEL_READ)) {
                    int32_t vel_raw = groupSyncRead_->getData(id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY);
                    int32_t pos_raw = groupSyncRead_->getData(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);

                    double vel_rad_s = (double)vel_raw * VELOCITY_UNIT_RPM * RPM_TO_RAD_SEC;
                    double pos_deg = (double)pos_raw * 0.088 - offsets_[i];
                    double pos_rad = pos_deg * (M_PI / 180.0);

                    // Shared Memory에 쓰기
                    state_ptr_[0 * 12 + i] = pos_rad;  // row 0: position
                    state_ptr_[1 * 12 + i] = vel_rad_s; // row 1: velocity

                    // ROS Topic 발행용 데이터 추가
                    msg.name.push_back(joint_names_[i]);
                    msg.position.push_back(pos_rad);
                    msg.velocity.push_back(vel_rad_s);
                }
            }
            pub_joint_state_->publish(msg);
        }

        // 2. READ from SHM (Command) & WRITE to Dynamixel
        groupSyncWriteCurrent_->clearParam();
        for (size_t i = 0; i < target_ids_.size(); ++i) {
            uint8_t id = target_ids_[i];
            
            // 바닥 모터(1, 5, 9)는 위치 제어이므로 전류 명령에서 제외 (필요 시 로직 추가)
            if (id == 1 || id == 5 || id == 9) continue;

            double torque_nm = cmd_ptr_[i]; // Python 노드가 쓴 값을 직접 읽음
            double current_a = torque_nm / KT_CONSTANT;
            int16_t goal_current = (int16_t)(current_a / CURRENT_UNIT_A);

            // Safety Limit
            if (goal_current > 600) goal_current = 600;
            if (goal_current < -600) goal_current = -600;

            uint8_t param[2];
            param[0] = DXL_LOBYTE(goal_current);
            param[1] = DXL_HIBYTE(goal_current);
            groupSyncWriteCurrent_->addParam(id, param);
        }
        groupSyncWriteCurrent_->txPacket();
    }

    // Members
    std::vector<uint8_t> target_ids_;
    std::vector<double> offsets_;
    std::vector<std::string> joint_names_;

    // Shared Memory FDs and Pointers
    int shm_state_fd_, shm_cmd_fd_;
    double *state_ptr_, *cmd_ptr_;

    // ROS 2
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joint_state_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Dynamixel SDK
    dynamixel::PortHandler *portHandler_;
    dynamixel::PacketHandler *packetHandler_;
    dynamixel::GroupSyncRead *groupSyncRead_;
    dynamixel::GroupSyncWrite *groupSyncWriteCurrent_;
    dynamixel::GroupSyncWrite *groupSyncWritePosition_;
    dynamixel::GroupSyncWrite *groupSyncWriteTorque_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DynamixelSHMNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}