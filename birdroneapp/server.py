import torch
import numpy as np
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from PIL import Image
import io
import cv2
import torchvision
import os
from ultralytics import YOLO
from torchvision.transforms import functional as F
from deep_sort_realtime.deepsort_tracker import DeepSort  # Import DeepSort from the deepsort package
import sys
import logging  # Import logging module
import warnings  # Import warnings module
# sys.path.append('D:/project/code')
from utils.video_utils import read_video, save_video

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load YOLO model
yolo_model = YOLO("F://Udaykiran//FinalYearProject//BirDrone//models//best_100_epoch.pt")

# Load Faster R-CNN model
num_classes = 3  # background + bird + drone
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
faster_rcnn_model.load_state_dict(torch.load('F://Udaykiran//FinalYearProject//BirDrone//models//fasterrcnn_resnet50_epoch_100.pth', map_location=device))
faster_rcnn_model.to(device)
faster_rcnn_model.eval()

# Initialize the DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.4, nn_budget=None)

TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp_videos')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Dictionary to keep track of active processing tasks
active_tasks = {}

def detect_objects_with_yolo(image):
    yolo_results = yolo_model(image)
    yolo_detections = yolo_results[0].boxes.data.cpu().numpy()
    yolo_classes = yolo_results[0].names

    detections = []
    for detection in yolo_detections:
        x1, y1, x2, y2, conf, cls_id = detection
        cls_id = int(cls_id)
        class_name = yolo_classes[cls_id]
        detections.append({
            'label': class_name,
            'confidence': float(conf),  # Convert to float
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'color': (0, 255, 0) if (class_name == "bird" or class_name=="birds") else (0, 0, 255)
        })
    return detections

def detect_objects_with_faster_rcnn(image_array):
    image = Image.fromarray(image_array)
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = faster_rcnn_model(image_tensor)

    detections = []
    class_names = {0: 'background', 1: 'bird', 2: 'drone'}
    for i in range(len(predictions[0]['boxes'])):
        confidence = predictions[0]['scores'][i].item()
        if confidence > 0.5:
            box = predictions[0]['boxes'][i].cpu().numpy()
            label = predictions[0]['labels'][i].item()
            bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            label_name = class_names.get(label, f"class_{label}")
            detections.append({
                'label': label_name,
                'confidence': float(confidence),  # Convert to float
                'bbox': bbox,
                'color': (0, 255, 0) if (label_name == "bird"or label_name=="birds") else (0, 0, 255)
            })
    return detections

def process_image(image_array):
    yolo_detections = detect_objects_with_yolo(image_array)
    final_detections = []

    for det in yolo_detections:
        if det['confidence'] < 0.5:
            faster_rcnn_detections = detect_objects_with_faster_rcnn(image_array)
            final_detections.extend(faster_rcnn_detections)
        else:
            final_detections.append(det)

    if not final_detections:
        final_detections = detect_objects_with_faster_rcnn(image_array)

    return final_detections

def calculate_precision_and_accuracy(detections):
    if not detections:
        return 0.0, 0.0

    total_detections = len(detections)
    high_confidence_detections = len([d for d in detections if d['confidence'] > 0.7])
    precision = high_confidence_detections / total_detections if total_detections > 0 else 0.0
    total_confidence = sum(float(d['confidence']) for d in detections)  # Convert to float
    accuracy = total_confidence / total_detections if total_detections > 0 else 0.0

    return precision, accuracy

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        file_extension = os.path.splitext(file.filename)[1].lower()
        logging.debug(f"Processing file with extension: {file_extension}")

        # Log the received file details
        file_content = file.read()
        logging.debug(f"Received file: {file.filename}, size: {len(file_content)} bytes")
        file.seek(0)  # Reset file pointer after reading

        # Get the target selection from the form data ("bird", "drone", or "both")
        selected_target = request.form.get('target', None)
        task_id = request.form.get('task_id', None)

        if not task_id:
            return jsonify({'error': 'No task ID provided'}), 400

        active_tasks[task_id] = True

        if file_extension in ['.jpg', '.jpeg', '.png','.webp']:
            image = Image.open(io.BytesIO(file_content)).convert('RGB')
            image_array = np.array(image)
            detections = process_image(image_array)

            # Filter detections only if target is "bird" or "drone"
            if selected_target == 'bird':
                detections = [det for det in detections if det['label'] == 'bird']
            elif selected_target == 'drone':
                detections = [det for det in detections if det['label'] == 'drone']
            # If "both", we do not filter

            precision, accuracy = calculate_precision_and_accuracy(detections)
            img_with_boxes = draw_bounding_boxes(image_array.copy(), detections, precision, accuracy)

            if not active_tasks.get(task_id):
                return jsonify({'error': 'Task was cancelled'}), 400

            return jsonify({
                'detections': [{**det, 'confidence': float(det['confidence'])} for det in detections],
                'precision': float(precision),  # Convert to float
                'accuracy': float(accuracy),  # Convert to float
                'image_with_boxes': img_with_boxes,
                'type': 'image'
            })

        elif file_extension in ['.mp4', '.avi', '.mov', '.webm']:
            temp_input_path = os.path.join(TEMP_DIR, f'input_{file.filename}')
            temp_output_path = os.path.join(TEMP_DIR, f'output_{file.filename}')
            
            try:
                with open(temp_input_path, 'wb') as f:
                    f.write(file_content)
                
                frames = read_video(temp_input_path)
                
                processed_frames = []

                for frame in frames:
                    if not active_tasks.get(task_id):
                        return jsonify({'error': 'Task was cancelled'}), 400

                    detections = process_image(frame)

                    if selected_target == 'bird':
                        detections = [det for det in detections if det['label'] == 'bird' or det['label'=='birds'] ]
                    elif selected_target == 'drone':
                        detections = [det for det in detections if det['label'] == 'drone' or det['label'=='drones']]
                    # If "both", no filter

                    frame_with_boxes = frame.copy()
                    draw_bounding_boxes(frame_with_boxes, detections, None, None)
                    processed_frames.append(frame_with_boxes)

                save_video(processed_frames, temp_output_path)

                if not active_tasks.get(task_id):
                    return jsonify({'error': 'Task was cancelled'}), 400

                return send_file(temp_output_path, as_attachment=True)

            except Exception as e:
                logging.error(f"Error processing video: {e}")
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                raise e

        return jsonify({'error': 'Unsupported file format'}), 400

    except Exception as e:
        logging.error(f"Failed to process file: {e}")
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 500

@socketio.on('cancel_task')
def handle_cancel_task(data):
    task_id = data.get('task_id')
    if task_id and task_id in active_tasks:
        active_tasks[task_id] = False
        emit('task_cancelled', {'task_id': task_id})

def draw_bounding_boxes(image_array, detections, precision, accuracy):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(image_array, (x1, y1), (x2, y2), det['color'], 2)
        cv2.putText(image_array, det['label'], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, det['color'], 2)

    # Convert BGR back to RGB before encoding
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

if __name__ == '__main__':
    socketio.run(app, debug=True)
