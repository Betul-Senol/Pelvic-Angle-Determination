import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Model yolu
model_path = "pelvis_asıl.pt"
model = YOLO(model_path)

# Test görüntüsünün yolu
test_image_path = "Test_Resimleri/477.jpg"

# Sonuçları kaydetmek için dizin
save_dir = "results/"
os.makedirs(save_dir, exist_ok=True)

# Görüntüyü yükle
image = cv2.imread(test_image_path)

results = model.predict(source=test_image_path, save=False, conf=0.2)

def calculate_outer_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    angle_deg = np.degrees(angle)
    if angle_deg < 180:
        angle_deg = 180 - angle_deg
    return angle_deg

def calculate_inner_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    angle_deg = np.degrees(angle)
    return angle_deg


img_with_circles = cv2.imread(test_image_path)
pelvic_point = None
suorcil_point = None

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        label = result.names[cls]

        if label == "FEMUR":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if x1 > img_with_circles.shape[1] // 2:
                continue

            femur_head_center_x = x2
            femur_head_center_y = y1 + (y2 - y1) // 4

            radius_x = (x2 - x1) // 4
            radius_y = (y2 - y1) // 4
            radius = min(radius_x, radius_y)
            radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

            femur_head_center_x = min(femur_head_center_x, x2 - radius)
            femur_head_center_y = min(femur_head_center_y, y1 + radius)

            cv2.circle(img_with_circles, (femur_head_center_x, femur_head_center_y), radius, (255, 0, 0), 3)
            cv2.circle(img_with_circles, (femur_head_center_x, femur_head_center_y), 5, (0, 255, 0), -1)

            offset = 20
            pelvic_point = (int((x1 + x2) / 2) + offset, y1)

            top_point = (femur_head_center_x, femur_head_center_y - radius)
            cv2.line(img_with_circles, (femur_head_center_x, femur_head_center_y), top_point, (0, 255, 0), 2)

            femur_vector = np.array([0, femur_head_center_y - y1])
            pelvic_vector = np.array([pelvic_point[0] - femur_head_center_x, pelvic_point[1] - femur_head_center_y])

            lcea_angle = calculate_outer_angle(femur_vector, pelvic_vector)

        if label == "SUORCIL" and suorcil_point is None:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 > img_with_circles.shape[1] // 2:
                continue

            offset = 20
            suorcil_point = (x1, y1 + offset)

            cv2.circle(img_with_circles, suorcil_point, 5, (0, 0, 255), -1)
            cv2.rectangle(img_with_circles, (x1, y1), (x2, y2), (0, 0, 255), 2)

            suorcil_vector = np.array([suorcil_point[0] - femur_head_center_x, suorcil_point[1] - femur_head_center_y])
            suorcil_angle = calculate_outer_angle(femur_vector, suorcil_vector)

            cv2.line(img_with_circles, (femur_head_center_x, femur_head_center_y), suorcil_point, (0, 0, 255), 2)

suorcil_boxes = []
teardrop_boxes = []
suorcil_point2 = None
teardrop_point = None

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        label = result.names[cls]

        if label == "SUORCIL":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if x1 > img_with_circles.shape[1] // 2:
                overlap = False
                for (prev_x1, prev_y1, prev_x2, prev_y2) in suorcil_boxes:
                    if not (x2 < prev_x1 or x1 > prev_x2 or y2 < prev_y1 or y1 > prev_y2):
                        overlap = True
                        break

                if not overlap:
                    suorcil_rightmost_x = x2
                    suorcil_rightmost_y = y1 + (y2 - y1) // 2
                    cv2.circle(img_with_circles, (suorcil_rightmost_x, suorcil_rightmost_y), 5, (255, 255, 0), -1)
                    suorcil_point2 = (suorcil_rightmost_x, suorcil_rightmost_y)

        elif label == "TEARDROP":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if x1 > img_with_circles.shape[1] // 2:
                overlap = False
                for (prev_x1, prev_y1, prev_x2, prev_y2) in teardrop_boxes:
                    if not (x2 < prev_x1 or x1 > prev_x2 or y2 < prev_y1 or y1 > prev_y2):
                        overlap = True
                        break

                if not overlap:
                    teardrop_bottom_x = x1 + (x2 - x1) // 2
                    teardrop_bottom_y = y2
                    teardrop_top_y = y1
                    cv2.circle(img_with_circles, (teardrop_bottom_x, teardrop_top_y), 5, (0, 165, 255), -1)
                    teardrop_point = (teardrop_bottom_x, teardrop_bottom_y)

if suorcil_point2 and teardrop_point:
    cv2.line(img_with_circles, suorcil_point2, teardrop_point, (0, 255, 255), 2)


    cv2.line(img_with_circles, (teardrop_bottom_x, teardrop_top_y), (suorcil_rightmost_x, suorcil_rightmost_y), (255, 0, 255), 2)

    vector1 = np.array([suorcil_point2[0] - teardrop_point[0], suorcil_point2[1] - teardrop_point[1]])
    vector2 = np.array([1, 0])

    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)

    cosine_theta = dot_product / (magnitude_vector1 * magnitude_vector2)
    angle_radians = np.arccos(np.clip(cosine_theta, -1.0, 1.0))

    angle_degrees = np.degrees(angle_radians)

    print(f"Sharp Angle: {angle_degrees:.2f} derece")

    if suorcil_point2:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Sharp Açı: {angle_degrees:.2f}°"

        line_length = 700
        line_start_x = max(0, teardrop_point[0] - line_length // 2)
        line_end_x = min(img_with_circles.shape[1], teardrop_point[0] + line_length // 2)
        cv2.line(img_with_circles, (line_start_x, teardrop_point[1]), (line_end_x, teardrop_point[1]), (255, 0, 0), 2)

        cv2.line(img_with_circles, (line_start_x, teardrop_top_y), (line_end_x, teardrop_top_y), (0, 255, 0), 2)

        horizontal_vector = np.array([1, 0])
        new_line_vector = np.array([suorcil_rightmost_x - teardrop_bottom_x, suorcil_rightmost_y - teardrop_top_y])
        inner_angle_degrees = calculate_inner_angle(horizontal_vector, new_line_vector)


        cv2.putText(img_with_circles, f"LCEA: {lcea_angle:.2f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(img_with_circles, f"Sharp: {angle_degrees:.2f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(img_with_circles, f"Tonnis: {inner_angle_degrees:.2f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2
                    )

    output_path = os.path.join(save_dir, f"result_{os.path.basename(test_image_path)}")
    cv2.imwrite(output_path, img_with_circles)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    print("Model test işlemi tamamlandı. Sonuçlar kaydedildi: ", save_dir)
