import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.spatial.distance import cdist

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.players = {}
        self.next_id = 1
        self.max_disappeared = 10

    def detect_players(self, frame):
        results = self.model(frame)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()

                        if confidence > 0.5:
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'center': [center_x, center_y],
                                'confidence': confidence
                            }
                            detections.append(detection)
        return detections

    def extract_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(512)
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return hist.flatten()

    def match_players(self, detections, frame):
        if not self.players:
            for detection in detections:
                self.players[self.next_id] = {
                    'center': detection['center'],
                    'features': self.extract_features(frame, detection['bbox']),
                    'bbox': detection['bbox'],
                    'disappeared': 0,
                    'active': True
                }
                detection['id'] = self.next_id
                self.next_id += 1
            return detections

        active_players = {pid: p for pid, p in self.players.items() if p['active']}
        detection_centers = np.array([d['center'] for d in detections])
        player_centers = np.array([p['center'] for p in active_players.values()])

        if len(detection_centers) == 0 or len(player_centers) == 0:
            return detections

        distance_matrix = cdist(detection_centers, player_centers)
        matched = []
        used_players = set()

        for i, detection in enumerate(detections):
            best_pid = None
            best_score = float('inf')
            current_feat = self.extract_features(frame, detection['bbox'])

        for j, pid in enumerate(active_players.keys()):
            if pid in used_players:
                continue

            dist = distance_matrix[i][j]
            player_feat = self.players[pid]['features']
            sim = 1 - np.corrcoef(current_feat, player_feat)[0, 1] if not np.isnan(np.corrcoef(current_feat, player_feat)[0, 1]) else 1

            score = dist + sim * 100  # weight histogram difference
            if score < best_score and dist < 100:
                best_score = score
                best_pid = pid

            if best_pid is not None:
                self.players[best_pid]['center'] = detection['center']
                self.players[best_pid]['features'] = self.extract_features(frame, detection['bbox'])
                self.players[best_pid]['bbox'] = detection['bbox']
                self.players[best_pid]['disappeared'] = 0
                detection['id'] = best_pid
                used_players.add(best_pid)
            matched.append(detection)

        for pid in active_players:
            if pid not in used_players:
                self.players[pid]['disappeared'] += 1
                if self.players[pid]['disappeared'] > self.max_disappeared:
                    self.players[pid]['active'] = False

        return matched

    def try_reidentify(self, detection, frame):
        current_features = self.extract_features(frame, detection['bbox'])
        inactive_players = {pid: p for pid, p in self.players.items() if not p['active']}

        best_match = None
        best_similarity = 0

        for pid, info in inactive_players.items():
            similarity = np.corrcoef(current_features, info['features'])[0, 1]
            if not np.isnan(similarity) and similarity > best_similarity and similarity > 0.7:
                best_similarity = similarity
                best_match = pid

        if best_match is not None:
            self.players[best_match]['active'] = True
            self.players[best_match]['center'] = detection['center']
            self.players[best_match]['features'] = current_features
            self.players[best_match]['bbox'] = detection['bbox']
            self.players[best_match]['disappeared'] = 0
            return best_match

        return None