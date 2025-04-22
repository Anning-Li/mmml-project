import os
import cv2
import numpy as np
import torch
import math
from transformers import AutoProcessor
from transformers import RTDetrForObjectDetection, VitPoseForPoseEstimation
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)

pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device)

# Keypoint visualization config
palette = np.array([
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
    [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
    [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
])
keypoint_edges = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (5, 6), (11, 13), (13, 15), (12, 14), (14, 16), (11, 12)
]
link_colors = palette[[0] * 4 + [7] * 2 + [7] * 2 + [9] * 6 + [16] * 4]
keypoint_colors = palette[[16] * 5 + [9] * 6 + [0] * 6]

def draw_points(image, keypoints, scores, keypoint_colors, keypoint_score_threshold=0.3, radius=4):
    for kid, (kpt, kpt_score) in enumerate(zip(keypoints, scores)):
        if kpt_score > keypoint_score_threshold:
            color = tuple(int(c) for c in keypoint_colors[kid % len(keypoint_colors)])
            cv2.circle(image, (int(kpt[0]), int(kpt[1])), radius, color, -1)

def draw_links(image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold=0.3, stick_width=2):
    height, width, _ = image.shape
    for sk_id, sk in enumerate(keypoint_edges):
        x1, y1, score1 = int(keypoints[sk[0]][0]), int(keypoints[sk[0]][1]), scores[sk[0]]
        x2, y2, score2 = int(keypoints[sk[1]][0]), int(keypoints[sk[1]][1]), scores[sk[1]]
        if all(0 < v < width for v in [x1, x2]) and all(0 < v < height for v in [y1, y2]) and score1 > keypoint_score_threshold and score2 > keypoint_score_threshold:
            color = tuple(int(c) for c in link_colors[sk_id % len(link_colors)])
            length = ((y1 - y2)**2 + (x1 - x2)**2) ** 0.5
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            polygon = cv2.ellipse2Poly(((x1 + x2)//2, (y1 + y2)//2), (int(length//2), stick_width), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(image, polygon, color)

def process_folder(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    os.makedirs("keypoints_npy", exist_ok=True)
    os.makedirs("keypoints_line_maps", exist_ok=True)
    os.makedirs("keypoints_overlays", exist_ok=True)

    for fname in tqdm(image_files, desc=f"Processing {image_folder}"):
        fpath = os.path.join(image_folder, fname)
        image = cv2.imread(fpath)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Object detection
        inputs = person_image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = person_model(**inputs)
        target_sizes = torch.tensor([(image.shape[0], image.shape[1])])
        results = person_image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]
        person_boxes = results["boxes"][results["labels"] == 0].cpu().numpy()

        if len(person_boxes) == 0:
            continue

        # Convert to (x, y, w, h)
        person_boxes[:, 2] -= person_boxes[:, 0]
        person_boxes[:, 3] -= person_boxes[:, 1]

        # Keypoint detection
        pose_inputs = pose_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            pose_outputs = pose_model(**pose_inputs)
        pose_results = pose_processor.post_process_pose_estimation(pose_outputs, boxes=[person_boxes])

        if not pose_results or not isinstance(pose_results[0], list) or len(pose_results[0]) == 0:
            continue

        image_pose_result = pose_results[0]

        # Visualizations
        line_map = np.ones_like(image) * 255
        overlay_image = image.copy()
        all_keypoints = []

        for pose_result in image_pose_result:
            keypoints = np.array(pose_result["keypoints"])
            scores = np.array(pose_result["scores"])
            all_keypoints.append(np.concatenate([keypoints, scores[:, None]], axis=1))  # [x, y, conf]

            draw_links(line_map, keypoints, scores, keypoint_edges, link_colors)
            draw_links(overlay_image, keypoints, scores, keypoint_edges, link_colors)
            draw_points(line_map, keypoints, scores, keypoint_colors)
            draw_points(overlay_image, keypoints, scores, keypoint_colors)

        name, ext = os.path.splitext(fname)

        # Save .npy
        np.save(os.path.join("keypoints_npy", f"{name}.npy"), np.array(all_keypoints))

        # Save line map
        cv2.imwrite(os.path.join("keypoints_line_maps", f"{name}.jpg"), cv2.cvtColor(line_map, cv2.COLOR_RGB2BGR))

        # Save overlay
        cv2.imwrite(os.path.join("keypoints_overlays", f"{name}.jpg"), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

# Run on both folders
process_folder("train_images")
process_folder("test_images")
