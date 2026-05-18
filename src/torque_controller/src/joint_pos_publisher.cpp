#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "dynamixel_sdk/dynamixel_sdk.h"

using namespace std::chrono_literals;

// --- 설정 (Configuration) ---
#define ADDR_PRESENT_POSITION       132
#define LEN_PRESENT_POSITION        4
#define PROTOCOL_VERSION            2.0
#define BAUDRATE                    57600
#define DEVICENAME                  "/dev/ttyUSB0"

class DynamixelPublisher : public rclcpp::Node
{
public:
    DynamixelPublisher()
    : Node("dynamixel_publisher")
    {
        // 1. Publisher 설정
        publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("hand_joint_pos", 10);

        // 2. 오프셋 설정 (단위: Degree) - ID 1 ~ 12
        offsets_ = {
            160.0, 180.0, 180.0, 180.0,
            160.0, 180.0, 180.0, 180.0,
            200.0, 180.0, 180.0, 180.0
        };

        // 3. Dynamixel 핸들러 초기화
        portHandler_ = dynamixel::PortHandler::getPortHandler(DEVICENAME);
        packetHandler_ = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);
        groupSyncRead_ = new dynamixel::GroupSyncRead(portHandler_, packetHandler_, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);

        // 4. 포트 열기 및 보드레이트 설정
        if (portHandler_->openPort()) {
            RCLCPP_INFO(this->get_logger(), "Succeeded to open the port");
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to open the port!");
            rclcpp::shutdown();
            return;
        }

        if (portHandler_->setBaudRate(BAUDRATE)) {
            RCLCPP_INFO(this->get_logger(), "Succeeded to change the baudrate");
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to change the baudrate!");
            rclcpp::shutdown();
            return;
        }

        // 5. 모터 스캔 (Ping)
        scan_motors();

        // 6. 타이머 설정 (50Hz = 20ms)
        timer_ = this->create_wall_timer(
            20ms, std::bind(&DynamixelPublisher::timer_callback, this));
    }

    virtual ~DynamixelPublisher()
    {
        portHandler_->closePort();
        delete groupSyncRead_;
        // SDK의 Port/Packet Handler는 싱글톤이 아니므로 delete 필요할 수 있으나 
        // 보통 OS가 프로세스 종료 시 정리함. 명시적 해제는 생략하거나 delete 수행.
    }

private:
    void scan_motors()
    {
        uint8_t dxl_error = 0;
        uint16_t dxl_model_number;
        int dxl_comm_result = COMM_TX_FAIL;

        RCLCPP_INFO(this->get_logger(), "Scanning motors (ID 1-12)...");

        for(uint8_t id = 1; id <= 12; id++) {
            dxl_comm_result = packetHandler_->ping(portHandler_, id, &dxl_model_number, &dxl_error);
            
            if (dxl_comm_result == COMM_SUCCESS) {
                // Ping 성공 시 SyncRead 목록에 추가
                if (groupSyncRead_->addParam(id)) {
                    target_ids_.push_back(id);
                    RCLCPP_INFO(this->get_logger(), "[ID:%02d] Found & Added.", id);
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "[ID:%02d] Not Found.", id);
            }
        }
        RCLCPP_INFO(this->get_logger(), "Scan Complete. Total %ld motors.", target_ids_.size());
    }

    void timer_callback()
    {
        // 1. SyncRead 패킷 전송
        int dxl_comm_result = groupSyncRead_->txRxPacket();
        
        // 통신 실패 시 로그 없이 리턴 (터미널 스팸 방지)
        if (dxl_comm_result != COMM_SUCCESS) {
            return; 
        }

        // 2. 메시지 준비
        auto msg = std_msgs::msg::Float64MultiArray();
        
        // ID 1부터 12까지 순회하며 데이터 채우기
        for (int i = 0; i < 12; i++) {
            uint8_t id = i + 1;
            double final_deg = 0.0;
            double offset = offsets_[i];

            // 해당 ID가 연결되어 있고 데이터가 사용 가능한지 확인
            bool is_available = false;
            // vector에 id가 있는지 확인 (간단한 검색)
            for(int tid : target_ids_) { if(tid == id) is_available = true; }

            if (is_available && groupSyncRead_->isAvailable(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)) {
                // Raw Data 읽기 (uint32_t)
                uint32_t present_pos_raw = groupSyncRead_->getData(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);
                
                // C++에서는 int32_t로 캐스팅하면 2's complement 음수 처리가 자동으로 됨
                int32_t present_pos_signed = (int32_t)present_pos_raw;

                // Degree 변환 (0.088 deg/pulse)
                double current_deg = present_pos_signed * 0.088;
                
                final_deg = current_deg - offset;
            } else {
                // 모터가 없거나 읽기 실패 시 Offset만 보냄 (0도 + Offset)
                final_deg = 0;
            }

            msg.data.push_back(final_deg * M_PI / 180.0);
        }

        // 3. 발행
        publisher_->publish(msg);
    }

    // 멤버 변수
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    dynamixel::PortHandler *portHandler_;
    dynamixel::PacketHandler *packetHandler_;
    dynamixel::GroupSyncRead *groupSyncRead_;

    std::vector<uint8_t> target_ids_;
    std::vector<double> offsets_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynamixelPublisher>());
    rclcpp::shutdown();
    return 0;
}