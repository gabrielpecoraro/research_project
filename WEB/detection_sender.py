from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import requests

# Load models
weapon_model = YOLO("best.pt")  # Replace with your trained model
person_model = YOLO("yolov8n.pt")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Setup
os.makedirs("criminal_logs", exist_ok=True)
cap = cv2.VideoCapture(0)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

# Timing control
last_capture_time = 0
capture_interval = 5  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Capture only every 5 seconds
    if current_time - last_capture_time >= capture_interval:
        last_capture_time = current_time

        # Run detection
        weapon_results = weapon_model(frame)[0]
        person_results = person_model(frame)[0]

        weapon_boxes = weapon_results.boxes.data.cpu().numpy()
        person_boxes = person_results.boxes.data.cpu().numpy()

        for person in person_boxes:
            x1, y1, x2, y2, conf, cls = person
            if int(cls) != 0:
                continue  # Skip if not person

            # Check for weapon proximity using IoU
            has_weapon = any(
                compute_iou([x1, y1, x2, y2], [w[0], w[1], w[2], w[3]]) > 0.2
                for w in weapon_boxes
            )

            if has_weapon:
                person_img = frame[int(y1):int(y2), int(x1):int(x2)]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"criminal_logs/suspect_{timestamp}.jpg"
                cv2.imwrite(filename, person_img)

                # Generate description
                pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                inputs = processor(pil_img, return_tensors="pt")
                out = caption_model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)

                # Send to FastAPI server
                try:
                    with open(filename, "rb") as f:
                        res = requests.post("http://localhost:8000/api/report", data={
                            "timestamp": timestamp,
                            "description": caption
                        }, files={"image": f})
                    print(f"[SENT] {timestamp} - {caption}")
                except Exception as e:
                    print("Failed to send report:", e)

    # Always show the most recent frame
    display_frame = frame.copy()
    cv2.imshow("Criminal Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
