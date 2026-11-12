import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import deque
import winsound
import time

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

print("\n=== Digital Screen Scanner (Noise-Filtered Edition) is Running! ===")
print("- Press 'c' to clear sequence.")
print("- Press 'q' to exit.\n")

sequence_history = deque(maxlen=20)
persistent_sequence = ""
current_confidence = 0.0
locked_radar_boxes = []
missing_frames_counter = 0

captured_full_sequence = ""
last_appended_digit_time = 0

total_detections_count = 0
confidence_history_sum = 0.0
confidence_samples_count = 0

prev_time = time.time()
fps_smooth = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    
    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time
    if time_diff > 0:
        instant_fps = 1.0 / time_diff
        fps_smooth = (fps_smooth * 0.8) + (instant_fps * 0.2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_sequence = ""
    current_frame_radar_boxes = []
    frame_max_conf = 0.0

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
                conf_score = 0.0
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
                    
                    if pred_digit in [1, 7]:
                        if aspect_w_h < 0.33:
                            pred_digit = 1
                        elif aspect_w_h > 0.55:
                            pred_digit = 7
                        else:
                            h_crop, w_crop = digit_cropped.shape
                            top_half = digit_cropped[:h_crop//3, :]
                            top_density = np.sum(top_half > 0) / float(top_half.size) if top_half.size > 0 else 0
                            if top_density > 0.35:
                                pred_digit = 7
                            else:
                                pred_digit = 1

                    # تخزين نسبة الثقة والرقم في البوكس مع تمرير الـ score
                    if conf_score > 55:
                        detected_sequence += str(pred_digit)
                        frame_max_conf = max(frame_max_conf, conf_score)

                current_frame_radar_boxes.append((x_start, y_start, x_end, y_end, str(pred_digit if pred_digit != -1 else ""), conf_score))

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
            current_confidence = 0.0

    if len(sequence_history) > 0:
        stable_sequence = max(set(sequence_history), key=sequence_history.count)
        if stable_sequence != persistent_sequence:
            persistent_sequence = stable_sequence
            total_detections_count += 1
            
            if current_time - last_appended_digit_time > 0.8:
                captured_full_sequence += persistent_sequence
                last_appended_digit_time = current_time

            try:
                winsound.Beep(1000, 150)
            except:
                pass
        if frame_max_conf > 0:
            current_confidence = frame_max_conf
            confidence_history_sum += current_confidence
            confidence_samples_count += 1

    # === رسم البوكسات المتوهجة فقط للكونتورات ذات الثقة العالية (> 45) ===
    active_boxes = current_frame_radar_boxes if current_frame_radar_boxes else locked_radar_boxes
    
    if active_boxes:
        overlay_glow = display_frame.copy()
        for box_data in active_boxes:
            if len(box_data) == 6:
                x_s, y_s, x_e, y_e, p_digit, c_score = box_data
            else:
                x_s, y_s, x_e, y_e, p_digit = box_data
                c_score = 100.0 # للبوكسات القديمة المحفوظة

            # فلتر منع إزعاج الحيط: لا ترسم إلا إيلا كانت الثقة زايدة على 45%
            if c_score > 51 and p_digit != "":
                cv2.rectangle(overlay_glow, (x_s - 4, y_s - 4), (x_e + 4, y_e + 4), (0, 255, 100), 4)
                cv2.rectangle(overlay_glow, (x_s - 2, y_s - 2), (x_e + 2, y_e + 2), (0, 255, 160), 2)
                cv2.rectangle(display_frame, (x_s, y_s), (x_e, y_e), (0, 255, 220), 2)
                
                cv2.rectangle(display_frame, (x_s, max(5, y_s - 28)), (x_s + 35, y_s), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (x_s, max(5, y_s - 28)), (x_s + 35, y_s), (0, 255, 220), 1)
                cv2.putText(display_frame, p_digit, (x_s + 8, max(22, y_s - 7)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 220), 2)

        cv2.addWeighted(overlay_glow, 0.35, display_frame, 0.65, 0, display_frame)

    # === الواجهة الرئيسية في الجهة اليسرى (Detected Digit & Progress Bar HUD) ===
    if persistent_sequence:
        ui_text = f"DETECTED: [ {persistent_sequence} ]"
        conf_text = f"CONF: {current_confidence:.1f}%"
        
        x1, y1, x2, y2 = 20, 20, 440, 105
        sub_img = display_frame[y1:y2, x1:x2]
        glass_overlay = np.full(sub_img.shape, (15, 50, 25), dtype=np.uint8)
        blended_glass = cv2.addWeighted(sub_img, 0.50, glass_overlay, 0.50, 0)
        display_frame[y1:y2, x1:x2] = blended_glass
        
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 150), 2)
        cv2.rectangle(display_frame, (x1+2, y1+2), (x2-2, y2-2), (0, 100, 60), 1)
        
        cv2.putText(display_frame, ui_text, (x1 + 15, y1 + 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(display_frame, conf_text, (x1 + 305, y1 + 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 2)
        
        bar_x1, bar_y1 = x1 + 15, y1 + 54
        bar_x2, bar_y2 = x2 - 15, y1 + 76
        
        cv2.rectangle(display_frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (25, 25, 25), -1)
        
        fill_width = int((bar_x2 - bar_x1) * (current_confidence / 100.0))
        fill_width = max(0, min(fill_width, bar_x2 - bar_x1))
        
        if fill_width > 0:
            cv2.rectangle(display_frame, (bar_x1 + 2, bar_y1 + 3), (bar_x1 + fill_width - 2, bar_y2 - 3), (0, 255, 140), -1)
                
        cv2.rectangle(display_frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (120, 120, 120), 1)

    # === واجهة سلسلة الأرقام الطويلة في أسفل الشاشة (Multi-Digit Sequence HUD) ===
    if captured_full_sequence:
        h_f, w_f, _ = display_frame.shape
        seq_x1, seq_y1, seq_x2, seq_y2 = 20, h_f - 80, w_f - 20, h_f - 20
        
        sub_seq = display_frame[seq_y1:seq_y2, seq_x1:seq_x2]
        seq_overlay = np.full(sub_seq.shape, (20, 20, 40), dtype=np.uint8)
        blended_seq = cv2.addWeighted(sub_seq, 0.40, seq_overlay, 0.60, 0)
        display_frame[seq_y1:seq_y2, seq_x1:seq_x2] = blended_seq
        
        cv2.rectangle(display_frame, (seq_x1, seq_y1), (seq_x2, seq_y2), (255, 150, 0), 2)
        
        full_seq_text = f"MULTI-DIGIT SEQUENCE: {captured_full_sequence}"
        cv2.putText(display_frame, full_seq_text, (seq_x1 + 15, seq_y1 + 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        hint_text = "Press 'c' to clear"
        cv2.putText(display_frame, hint_text, (seq_x2 - 170, seq_y1 + 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # === لوحة الإحصائيات الحية في الجهة اليمنى (Compact Stats Dashboard) ===
    h_frame, w_frame, _ = display_frame.shape
    sx1, sy1, sx2, sy2 = w_frame - 180, 20, w_frame - 15, 105
    
    sub_stats = display_frame[sy1:sy2, sx1:sx2]
    stats_overlay = np.full(sub_stats.shape, (30, 30, 30), dtype=np.uint8)
    blended_stats = cv2.addWeighted(sub_stats, 0.45, stats_overlay, 0.55, 0)
    display_frame[sy1:sy2, sx1:sx2] = blended_stats
    
    cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), (0, 200, 255), 2)
    
    avg_conf = (confidence_history_sum / confidence_samples_count) if confidence_samples_count > 0 else 0.0
    
    fps_text = f"FPS: {int(fps_smooth)}"
    det_text = f"DET: {total_detections_count}"
    avg_text = f"AVG: {avg_conf:.1f}%"
    
    cv2.putText(display_frame, "SYS STATS", (sx1 + 8, sy1 + 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    cv2.putText(display_frame, fps_text, (sx1 + 8, sy1 + 39), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.putText(display_frame, det_text, (sx1 + 8, sy1 + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.putText(display_frame, avg_text, (sx1 + 8, sy1 + 81), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1)

    cv2.imshow('Digital Screen Scanner', display_frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('c'):
        captured_full_sequence = ""
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()