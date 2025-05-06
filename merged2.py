import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from flask import Flask, Response

# DeepSORT 관련 모듈 임포트 (DeepSORT 구현체 파일 경로에 따라 수정 필요)
# 예시: deep_sort 폴더가 현재 디렉토리에 있는 경우
from deep_sort.deep_sort import DeepSort

# --- 설정 값 ---
# SSD MobileNet 모델 파일 경로 (실제 파일 경로로 변경 필요)
PROTOTXT_PATH = 'MobileNetSSD_deploy.prototxt'
MODEL_PATH = 'MobileNetSSD_deploy.caffemodel'

# 클래스 이름 파일 경로 (실제 파일 경로로 변경 필요)
# 모델 학습 시 사용된 클래스 순서와 일치해야 함
CLASSES_PATH = 'coco_classes.txt'

# 객체 탐지 신뢰도 임계값
CONF_THRESHOLD = 0.4 # YOLO 보다 조금 낮게 시작하는 경우가 많음

# 웹캠 설정
CAMERA_DEVICE = "/dev/video4" # 실제 사용하는 카메라 장치 번호로 변경
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# GPIO 설정
GPIO.setmode(GPIO.BCM)
LED_PIN = 2
GPIO.setup(LED_PIN, GPIO.OUT)

# DeepSORT 설정
# Re-ID 모델 경로 (사용하지 않거나 다른 모델을 사용하는 경우 수정 또는 주석 처리)
# 라즈베리파이에서 Re-ID 모델 연산은 부담될 수 있습니다.
# DeepSORT 구현체에 따라 Re-ID 모델 경로가 필요 없을 수도 있습니다.
DEEPSORT_MODEL_PATH = 'mars-small128.pb' # 예시 경로

# --- 모델 및 추적기 로드 ---
# 1) SSD MobileNet 모델 로드
try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("SSD MobileNet model loaded successfully.")
except cv2.error as e:
    print(f"Error loading SSD MobileNet model: {e}")
    print("Please check PROTOTXT_PATH and MODEL_PATH.")
    exit()

# 클래스 이름 로드
try:
    with open(CLASSES_PATH, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(CLASSES)} classes.")
except FileNotFoundError:
    print(f"Error: Class names file not found at {CLASSES_PATH}")
    exit()

# 2) DeepSORT 추적기 초기화
try:
    # DeepSORT 초기화 방식은 사용하는 구현체에 따라 다를 수 있습니다.
    # 아래는 일반적인 예시이며, Re-ID 모델 사용 여부 등 설정이 필요할 수 있습니다.
    # metric="cosine"은 외형 정보(Re-ID 모델) 사용 시, metric="euclidean" 등은 사용하지 않을 때
    # max_iou_distance, max_age_milliseconds 등 파라미터는 필요에 따라 조정
    tracker = DeepSort(model_path=DEEPSORT_MODEL_PATH, max_dist=0.2, min_confidence=0.3,
                       nms_max_overlap=1.0, max_iou_distance=0.7, max_age_seconds=1.0, n_init=3, gid_alpha=0.5, n_batches=1, only_process_classes=None)
    print("DeepSORT tracker initialized successfully.")
except Exception as e:
    print(f"Error initializing DeepSORT tracker: {e}")
    print("Please ensure DeepSORT implementation files are correctly imported and Re-ID model path is valid if used.")
    exit()

# 3) 웹캠 열기
cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"Cannot open the camera device {CAMERA_DEVICE}")
    exit()

# Flask 앱 설정
app = Flask(__name__)

def generate_frames():
    """카메라에서 프레임을 캡처하여 SSD MobileNet 탐지, DeepSORT 추적 후, JPEG로 인코딩해 스트림 전달"""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            # 카메라 연결이 끊어졌을 가능성 있음, 적절한 에러 처리 필요
            break # 혹은 continue

        # 프레임 차원 가져오기
        (h, w) = frame.shape[:2]

        # SSD MobileNet 입력 blob 생성 및 추론
        # SSD MobileNet 모델의 입력 크기에 따라 300x300 등으로 변경될 수 있음
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # DeepSORT 입력 형식에 맞게 탐지 결과 가공 및 필터링
        # DeepSORT는 일반적으로 [x1, y1, w, h, confidence, class_id] 또는 [x1, y1, w, h, confidence] 형식을 기대
        # SSD MobileNet 출력 형식: [?, class_id, confidence, x_min, y_min, x_max, y_max]
        detection_list = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # 설정한 신뢰도 임계값보다 큰 탐지만 사용
            if confidence > CONF_THRESHOLD:
                # 바운딩 박스 좌표 계산 (원본 이미지 스케일로 복원)
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # DeepSORT 입력 형식 [x1, y1, w, h]으로 변환
                x1, y1 = startX, startY
                width, height = endX - startX, endY - startY

                # 클래스 ID 가져오기
                class_id = int(detections[0, 0, i, 1])
                # 클래스 이름 가져오기 (CLASSES 리스트와 class_id가 일치하는지 확인 필요)
                # SSD MobileNet의 class_id 0은 보통 'background'이므로 1부터 시작하는 모델 고려
                if class_id < len(CLASSES):
                    class_name = CLASSES[class_id]
                else:
                     class_name = "unknown" # 알 수 없는 클래스

                # DeepSORT에 전달할 형식으로 데이터 추가
                # 클래스 ID를 DeepSORT에 전달할 수 있는 구현체와 함께 사용하는 것이 추적 후 클래스 정보를 얻기 용이
                # 일부 DeepSORT 구현체는 class_id를 입력으로 받지 않을 수 있습니다.
                detection_list.append([x1, y1, width, height, confidence, class_id])

        # DeepSORT 추적 업데이트
        # detection_list는 numpy 배열 형태 [(x1, y1, w, h, confidence, class_id), ...] 가 될 수 있음
        detections_np = np.array(detection_list)

        # DeepSORT 추적기 업데이트 함수 호출
        # update 함수의 입력/출력 형식은 사용하는 DeepSORT 구현체에 따라 정확히 다를 수 있습니다.
        # 일반적으로 입력: 탐지 결과 (numpy 배열), 출력: 추적된 객체 목록 (트랙 정보 포함)
        tracked_objects = tracker.update(detections_np) # 혹은 tracker.update(detections_np[:, :5]) if class_id is not taken

        # 추적 결과 시각화 및 LED 제어
        # tracked_objects는 트랙 정보(예: 바운딩 박스, 트랙 ID, 클래스 ID 등)를 담고 있음
        # 정확한 형태는 사용하는 DeepSORT 구현체를 확인해야 합니다.
        # 일반적으로 각 트랙은 [x1, y1, x2, y2, track_id, class_id] 형태 정보를 가집니다.

        # 특정 클래스 객체가 추적 중인지 확인하는 플래그
        target_object_tracked = False

        for obj in tracked_objects:
            # obj 구조 예시: [x1, y1, x2, y2, track_id, class_id]
            # 사용하는 DeepSORT 구현체에 따라 obj의 구조가 다를 수 있습니다.
            # 특히 class_id 정보가 추적 결과에 포함되려면, DeepSORT 입력 시 class_id를 제공했거나
            # 탐지 모델의 class_id를 트랙과 연결하는 로직이 DeepSORT 내에 있어야 합니다.
            # 간단한 구현체는 [x1, y1, x2, y2, track_id]만 반환할 수 있습니다.

            bbox = obj[:4].astype(int)
            track_id = int(obj[4])
            # 클래스 정보가 추적 결과에 포함된 경우 사용
            if len(obj) > 5:
                 cls_id = int(obj[5])
                 if cls_id < len(CLASSES):
                     cls_name = CLASSES[cls_id]
                 else:
                      cls_name = "unknown"
            else:
                 cls_name = "N/A" # 클래스 정보 없음

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # 트랙 ID 및 클래스 이름 표시
            label = f"ID: {track_id} {cls_name}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # LED 제어 로직 (추적 중인 객체 클래스 확인)
            # 원본 코드와 같이 'car', 'motorcycle', 'bicycle' 클래스를 대상으로 함
            if cls_name in ['car', 'motorcycle', 'bicycle']:
                 target_object_tracked = True
                 # LED 제어는 추적 중인 객체가 발견되었을 때 한 번만 켜도록 수정 (과도한 깜빡임 방지)
                 # 만약 각 객체 탐지/추적 시마다 LED를 켜고 싶다면 아래 GPIO.output 라인을 사용하되, time.sleep 제거 필요
                 # GPIO.output(LED_PIN, GPIO.HIGH) # 객체 발견 시 LED 켜기 (짧게)

        # 추적 대상 객체가 하나라도 있으면 LED 켜기 (지속적으로)
        # 모든 추적 대상 객체가 사라지면 LED 끄기
        if target_object_tracked:
             GPIO.output(LED_PIN, GPIO.HIGH)
        else:
             GPIO.output(LED_PIN, GPIO.LOW)


        # JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue # 인코딩 실패 시 건너뛰기

        frame_bytes = buffer.tobytes()

        # MJPEG 스트림 포맷으로 yield
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # 프레임 스트림 종료 시 정리
    cap.release()
    # GPIO cleanup은 프로그램 종료 시 한 번만 호출하는 것이 좋습니다.
    # 여기서는 generate_frames 함수가 루프를 빠져나갈 때 호출되도록 했습니다.
    # Flask 애플리케이션 종료 시 GPIO를 정리하려면 별도의 핸들러를 추가해야 합니다.
    GPIO.cleanup()
    print("Camera released and GPIO cleaned up.")


@app.route('/video_feed')
def video_feed():
    """/video_feed URL로 MJPEG 스트림 전송"""
    # generate_frames에서 cleanup을 호출하므로 mimetype 뒤에 인자 추가 안 함
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """웹 브라우저 기본 페이지. video_feed를 iframe으로 임베딩"""
    return """
    <html>
      <head>
        <title>Real-time Object Tracking (SSD MobileNet + DeepSORT)</title>
      </head>
      <body>
        <h1>Live Camera Feed with Object Tracking</h1>
        <img src="/video_feed" width="640" height="480">
      </body>
    </html>
    """

if __name__ == "__main__":
    # Flask 앱 실행
    # 디버그 모드를 끄는 것이 라즈베리파이 성능에 유리합니다.
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        # Ctrl+C 등으로 종료 시 GPIO 정리
        print("Server stopped. Cleaning up GPIO.")
    finally:
        # 프로그램 종료 시 GPIO 정리 (KeyboardInterrupt 외의 종료에도 대비)
        GPIO.cleanup()
        print("Final GPIO cleanup.")