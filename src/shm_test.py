import multiprocessing.shared_memory as shm

try:
    # C++이 만든 공유 메모리 이름을 그대로 대입
    test_shm = shm.SharedMemory(name='target_pose_shm', create=False)
    print(f"✅ 호환성 확인 성공! 메모리 크기: {test_shm.size} 바이트")
except FileNotFoundError:
    print("❌ 형식 불일치 (FileNotFoundError): 파일은 존재하나 파이썬이 인식할 수 없는 이름 규격입니다.")
except Exception as e:
    print(f"⚠️ 기타 에러 발생: {e}")