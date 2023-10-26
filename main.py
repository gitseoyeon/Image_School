import os
from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)

# 경로 설정
coco_names = 'coco.names'  # 객체 이름 파일
yolov4_cfg = 'yolov4.cfg'  # YOLOv4 구성 파일
yolov4_weights = 'yolov4.weights'  # YOLOv4 가중치 파일

# YOLO 모델 초기화
net = cv2.dnn.readNetFromDarknet(yolov4_cfg, yolov4_weights)
output_layers = net.getUnconnectedOutLayersNames()
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# coco.names 파일을 읽어와서 클래스 레이블을 가져옴
with open(coco_names, 'r') as f:
    classes = f.read().strip().split('\n')

@app.route('/')
def index():
    return render_template('index.html')

# 이미지를 저장할 uploads 디렉토리 생성
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/upload', methods=['POST'])
def upload():
    # 업로드된 파일 가져오기
    uploaded_image = request.files['image']

    if uploaded_image:
        # 이미지 저장
        image_path = os.path.join('uploads', uploaded_image.filename)
        uploaded_image.save(image_path)

        # 이미지 분석
        results = analyze_image(image_path)

        return render_template('index.html', image_path=image_path, results=results)

    return "이미지를 업로드해주세요."


def analyze_image(image_path):
    # 이미지 불러오기
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # 이미지 YOLOv4로 분석
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 결과 저장
    class_ids = []
    confidences = []
    boxes = []
    results = []

    # 객체 식별
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # 객체의 중심 좌표, 너비, 높이 계산
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 객체의 좌상단 좌표 계산
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non-maximum suppression 적용
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in range(len(boxes)):
            if i in indexes:
                label = classes[class_id]
                confidence = confidences[i]
                x, y, w, h = boxes[i]

                results.append({'label': label, 'confidence': confidence, 'x': x, 'y': y, 'w': w, 'h': h})

    return results

if __name__ == '__main__':
    app.run(debug=True)
