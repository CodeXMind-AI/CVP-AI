import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import deque

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cpu')
model = SimpleCNN().to(device)

try:
    model.load_state_dict(torch.load('mnist_cnn_model.pth', map_location=device))
    print("=== Model weights loaded successfully! ===")
except Exception as e:
    print(f"Error loading weights: {e}")

model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

cap = cv2.VideoCapture(0)

print("\n=== Digital Scanner Vision (Strict Width Fix for 1 vs 7) is Running! ===")
print("- Press 'q' to exit.\n")

sequence_history = deque(maxlen=20)
persistent_sequence = ""
locked_radar_boxes = []
missing_frames_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_sequence = ""
    current_frame_radar_boxes = []

    if contours:
        raw_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if 400 < area < 12000:
                x, y, cw, ch = cv2.boundingRect(c)
                aspect_ratio = ch / float(cw) if cw > 0 else 0
                
                if 0.5 < aspect_ratio < 6.0:
                    box_area = cw * ch
                    extent = area / float(box_area) if box_area > 0 else 0
                    
                    if 0.15 < extent < 0.60:
                        aspect_w_h_box = cw / float(ch) if cw > 0 else 0
                        if aspect_w_h_box > 1.6 and cw > 40:
                            half_w = cw // 2
                            raw_candidates.append((x, y, half_w, ch, c))
                            raw_candidates.append((x + half_w, y, cw - half_w, ch, c))
                        else:
                            raw_candidates.append((x, y, cw, ch, c))

        if raw_candidates:
            raw_candidates = sorted(raw_candidates, key=lambda item: item[0])
            
            for (x, y, cw, ch, c) in raw_candidates:
                pad = 10
                h_f, w_f, _ = frame.shape
                x_start = max(0, x - pad)
                y_start = max(0, y - pad)
                x_end = min(w_f, x + cw + pad)
                y_end = min(h_f, y + ch + pad)
                
                digit_cropped = thresh[y_start:y_end, x_start:x_end]
                
                pred_digit = -1
                if digit_cropped.size > 0:
                    pil_img = Image.fromarray(digit_cropped)
                    img_tensor = transform(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(img_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)

                    pred_digit = predicted.item()
                    conf_score = confidence.item() * 100

                    aspect_w_h = cw / float(ch) if ch > 0 else 0
                    
                    # === الحل الصارم الجديد: الاعتماد على عرض البوكس والنسبة الهندسية المباشرة ===
                    if pred_digit in [1, 7]:
                        # إيلا كان الحرف نحيف بزاف (العرض مقارنة بالارتفاع صغير جداً) فهو 1 قطعاً
                        # أو إيلا كان العرض بالبيكسل (`cw`) صغير لدرجة مكيسمحش باش يكون 7
                        if aspect_w_h < 0.33:
                            pred_digit = 1
                        elif aspect_w_h > 0.55:
                            pred_digit = 7
                        else:
                            # فحص إضافي للتأكد من مركز الكتلة أو الكثافة العلوية بشكل أبسط
                            h_crop, w_crop = digit_cropped.shape
                            top_half = digit_cropped[:h_crop//3, :]
                            top_density = np.sum(top_half > 0) / float(top_half.size) if top_half.size > 0 else 0
                            
                            # الرقم 7 كيكون عامر في الفوق، بينما 1 كيكون متساوي
                            if top_density > 0.35:
                                pred_digit = 7
                            else:
                                pred_digit = 1

                    if conf_score > 55:
                        detected_sequence += str(pred_digit)

                current_frame_radar_boxes.append((x_start, y_start, x_end, y_end, str(pred_digit if pred_digit != -1 else "")))

    if detected_sequence:
        sequence_history.append(detected_sequence)
        locked_radar_boxes = current_frame_radar_boxes
        missing_frames_counter = 0
    else:
        missing_frames_counter += 1
        if missing_frames_counter > 150:
            sequence_history.clear()
            locked_radar_boxes = []
            persistent_sequence = ""

    if len(sequence_history) > 0:
        stable_sequence = max(set(sequence_history), key=sequence_history.count)
        persistent_sequence = stable_sequence

    active_boxes = current_frame_radar_boxes if current_frame_radar_boxes else locked_radar_boxes
    
    if active_boxes:
        for (x_s, y_s, x_e, y_e, p_digit) in active_boxes:
            cv2.rectangle(display_frame, (x_s, y_s), (x_e, y_e), (0, 255, 0), 2)
            cv2.rectangle(display_frame, (x_s-1, y_s-1), (x_e+1, y_e+1), (0, 255, 0), 1)
            if p_dir := p_digit:
                cv2.putText(display_frame, p_digit, (x_s, max(30, y_s - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if persistent_sequence:
        cv2.putText(display_frame, f"Scanner Number: {persistent_sequence}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow('Digital Screen Scanner', display_frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()