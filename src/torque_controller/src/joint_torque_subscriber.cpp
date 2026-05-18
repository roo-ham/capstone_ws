#include <chrono>
#include <memory>
#include <vector>
#include <cmath>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "dynamixel_sdk/dynamixel_sdk.h"

// --- Control Table (XH/XM Series) ---
#define ADDR_OPERATING_MODE         11
#define ADDR_TORQUE_ENABLE          64
#define ADDR_GOAL_CURRENT           102

#define LEN_TORQUE_ENABLE           1
#define LEN_GOAL_CURRENT            2

// Values
#define TORQUE_ENABLE               1
#define TORQUE_DISABLE              0
#define OP_MODE_CURRENT             0   // Current Control Mode
#define OP_MODE_POSITION            3   // Position Control Mode (Default)

// Communication
#define PROTOCOL_VERSION            2.0
#define BAUDRATE                    57600           
#define DEVICENAME                  "/dev/ttyUSB0" 

// Physics (Calibration required)
// XH430-W350: 1 unit ~ 2.69mA, Kt ~ 1.3 Nm/A (Ideal)
const double KT_CONSTANT = 1.0; 
const double CURRENT_UNIT_A = 0.00269; 

class JointTorqueSubscriber : public rclcpp::Node
{
public:
    JointTorqueSubscriber()
    : Node("joint_torque_subscriber"), torque_enabled_(false)
    {
        // 1. Dynamixel 초기화 (포트 오픈 -> 토크 끄기 -> 모드 변경)
        if (!init_dynamixel()) {
            RCLCPP_ERROR(this->get_logger(), "Dynamixel Initialization Failed!");
            rclcpp::shutdown();
            return;
        }

        // 2. Subscriber 설정
        subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "hand_joint_torque", 10,
            std::bind(&JointTorqueSubscriber::topic_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Ready! Operating Mode set to CURRENT(0). Waiting for commands...");
    }

    virtual ~JointTorqueSubscriber()
    {
        // 종료 시 토크 끄고 포트 닫기
        set_torque_all(false);
        portHandler_->closePort();
        delete groupSyncWriteCurrent_;
        delete groupSyncWriteTorque_;
    }

private:
    bool init_dynamixel()
    {
        portHandler_ = dynamixel::PortHandler::getPortHandler(DEVICENAME);
        packetHandler_ = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

        // 1. 포트 열기
        if (!portHandler_->openPort()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open port!");
            return false;
        }

        // 2. 보드레이트 설정
        if (!portHandler_->setBaudRate(BAUDRATE)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to set baudrate!");
            return false;
        }

        // 3. SyncWrite 인스턴스 생성
        groupSyncWriteCurrent_ = new dynamixel::GroupSyncWrite(
            portHandler_, packetHandler_, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT);
        
        groupSyncWriteTorque_ = new dynamixel::GroupSyncWrite(
            portHandler_, packetHandler_, ADDR_TORQUE_ENABLE, LEN_TORQUE_ENABLE);

        // ============================================================
        // [중요] 모드 변경 시퀀스: Torque OFF -> Set Mode -> Ready
        // ============================================================
        RCLCPP_INFO(this->get_logger(), "Configuring Dynamixels (ID 1-12)...");

        uint8_t dxl_error = 0;
        int dxl_comm_result = COMM_TX_FAIL;

        for (uint8_t id = 1; id <= 12; id++) {
            // A. Torque OFF (필수: 토크가 켜져 있으면 모드 변경 불가)
            dxl_comm_result = packetHandler_->write1ByteTxRx(
                portHandler_, id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
            
            if (dxl_comm_result != COMM_SUCCESS) {
                RCLCPP_WARN(this->get_logger(), "[ID:%d] Failed to disable torque.", id);
            }

            // B. Operating Mode 변경 (Address 11 -> 0: Current Control)
            dxl_comm_result = packetHandler_->write1ByteTxRx(
                portHandler_, id, ADDR_OPERATING_MODE, OP_MODE_CURRENT, &dxl_error);

            if (dxl_comm_result != COMM_SUCCESS) {
                RCLCPP_ERROR(this->get_logger(), "[ID:%d] Failed to set Current Mode!", id);
                return false; // 모드 설정 실패는 치명적 오류로 간주
            } else {
                 // RCLCPP_INFO(this->get_logger(), "[ID:%d] Set to Current Mode.", id);
            }
        }

        torque_enabled_ = false; // 내부 상태 플래그 초기화
        return true;
    }

    // 모든 모터 토크 ON/OFF 함수
    void set_torque_all(bool enable)
    {
        groupSyncWriteTorque_->clearParam();
        uint8_t data = enable ? TORQUE_ENABLE : TORQUE_DISABLE;
        uint8_t param[1] = { data };

        for (int i = 1; i <= 12; i++) {
            groupSyncWriteTorque_->addParam(i, param);
        }

        int dxl_comm_result = groupSyncWriteTorque_->txPacket();
        
        if (dxl_comm_result == COMM_SUCCESS) {
            torque_enabled_ = enable;
            RCLCPP_INFO(this->get_logger(), "Torque State Changed: %s", enable ? "ON" : "OFF");
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to switch torque state!");
        }
    }

    void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() != 12) return;

        // 1. 입력값이 모두 0인지 확인
        bool all_zeros = true;
        for (double val : msg->data) {
            if (std::abs(val) > 1e-6) {
                all_zeros = false;
                break;
            }
        }

        // 2. 상태 머신 (토크 ON/OFF 전환)
        if (all_zeros) {
            // 모두 0이고, 현재 토크가 켜져 있다면 -> 끈다
            if (torque_enabled_) {
                set_torque_all(false);
            }
            // 토크가 꺼져 있으면 전류 명령을 보낼 필요 없음
            return;
        } else {
            // 값이 들어왔는데, 현재 토크가 꺼져 있다면 -> 켠다
            if (!torque_enabled_) {
                set_torque_all(true);
            }
        }

        // 3. 전류 제어 명령 생성 및 전송
        groupSyncWriteCurrent_->clearParam();

        for (size_t i = 0; i < 12; i++) {
            uint8_t id = i + 1;
            double torque_nm = msg->data[i];

            // Torque(Nm) -> Current(A) -> DXL Value
            // Current = Torque / Kt
            double current_a = torque_nm / KT_CONSTANT;
            int16_t goal_current_val = (int16_t)(current_a / CURRENT_UNIT_A);

            // Safety Clamp (안전을 위해 +/- 2.0A 수준 제한 예시)
            // XH430 Max Current는 보통 1000~2000 사이 값이므로 적절히 조절
            int16_t limit = 600; 
            if (goal_current_val > limit) goal_current_val = limit;
            if (goal_current_val < -limit) goal_current_val = -limit;

            // 패킷 생성 (Low byte, High byte)
            uint8_t param[2];
            param[0] = DXL_LOBYTE(goal_current_val);
            param[1] = DXL_HIBYTE(goal_current_val);

            groupSyncWriteCurrent_->addParam(id, param);
        }

        groupSyncWriteCurrent_->txPacket();
    }

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
    
    dynamixel::PortHandler *portHandler_;
    dynamixel::PacketHandler *packetHandler_;
    dynamixel::GroupSyncWrite *groupSyncWriteCurrent_;
    dynamixel::GroupSyncWrite *groupSyncWriteTorque_;

    bool torque_enabled_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JointTorqueSubscriber>());
    rclcpp::shutdown();
    return 0;
}