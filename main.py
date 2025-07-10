import cv2
import os
from tracker import PlayerTracker

# === CONFIG ===
VIDEO_PATH = "15sec_input_720p.mp4"  # Input video file
MODEL_PATH = "best.pt"                # YOLOv11 model file
OUTPUT_PATH = "output/output_tracked.mp4"  # Output file path


def process_video(video_path, model_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = PlayerTracker(model_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = tracker.detect_players(frame)
        matched = tracker.match_players(detections, frame)

        # Try to reidentify any unmatched players
        for det in matched:
            if 'id' not in det:
                reid = tracker.try_reidentify(det, frame)
                if reid is not None:
                    det['id'] = reid
                else:
                    det['id'] = tracker.next_id
                    tracker.next_id += 1

        # Draw results
        for det in matched:
            x1, y1, x2, y2 = map(int, det['bbox'])
            pid = det['id']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    print(f"Video saved at: {output_path}")

    # Try to launch the video (Windows only)
    import time
    if os.name == 'nt' and os.path.exists(output_path):
        time.sleep(1)
        os.startfile(output_path)


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Video file not found: {VIDEO_PATH}")
        exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        exit(1)

    process_video(VIDEO_PATH, MODEL_PATH, OUTPUT_PATH)

import os
print(os.path.exists("output/output_tracked.mp4"))

