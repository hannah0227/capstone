import cv2
from ultralytics import YOLO
from flask import Flask, Response

app = Flask(__name__)

# 1) YOLO 모델 로드: 필요하면 'yolov5n.pt' 대신 'yolov5nu.pt'도 사용.
model = YOLO("yolov5n.pt")

# 2) 웹캠 혹은 RealSense RGB 노드 열기. (예: /dev/video? 번호 확인 후 사용)
#    (아래 예시는 /dev/video2로 가정. 필요시 숫자를 변경)
cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open the camera device")
    exit()

def generate_frames():
    """카메라에서 프레임을 캡처하여 YOLO 추론 후, JPEG로 인코딩해 스트림으로 전달"""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # 프레임을 정상적으로 받지 못했을 경우 건너뛰기

        # YOLO 추론 (임계값은 필요에 따라 조정)
        results = model(frame, conf=0.3)
        annotated_frame = results[0].plot()
        
        # (추가) 클래스에 따른 콘솔 출력 예시
        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls[0])]
            if   cls_name == 'car':
                print(0)
            elif cls_name == 'motorcycle':
                print(1)
            elif cls_name == 'bicycle':
                print(2)
        
        # JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # MJPEG 스트림 포맷으로 yield (multipart)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """/video_feed URL로 MJPEG 스트림 전송"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """웹 브라우저 기본 페이지. video_feed를 iframe으로 임베딩"""
    return """
    <html>
      <head>
        <title>Real-time YOLO Detection</title>
      </head>
      <body>
        <h1>Live Camera Feed with YOLO Detection</h1>
        <img src="/video_feed" width="640" height="480">
      </body>
    </html>
    """

if __name__ == "__main__":
    # 0.0.0.0으로 실행하여 네트워크 상의 모든 인터페이스에서 접속 가능하도록 함.
    app.run(host='0.0.0.0', port=5000, debug=False)
