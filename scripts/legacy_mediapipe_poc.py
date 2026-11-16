"""Stage 1 Proof-of-Concept: MediaPipe heuristic baseline."""

from collections import deque
import csv
import math
from pathlib import Path
import sys

import cv2
import mediapipe as mp

# Ensure project root is importable when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_INPUT_VIDEO,
    TELEMETRY_OUTPUT_DIR,
    VIDEOS_OUTPUT_DIR,
    ensure_directories,
)


from src.tracking.visualizer import create_video_writer


def main() -> None:
    ensure_directories()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    input_video_path = DEFAULT_INPUT_VIDEO
    output_video_path = VIDEOS_OUTPUT_DIR / "mediapipe_baseline_output.mp4"
    csv_report_path = TELEMETRY_OUTPUT_DIR / "mediapipe_baseline_report.csv"

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = create_video_writer(output_video_path, fps, width, height)

    wrist_history = deque(maxlen=fps)

    with open(csv_report_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Timestamp_sec", "Presence", "Posture", "Activity"])

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = round(frame_count / fps, 2)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            presence_state = "Away"
            posture_state = "-"
            activity_state = "-"

            if results.pose_landmarks:
                presence_state = "Present"
                landmarks = results.pose_landmarks.landmark

                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                # Posture Logic
                avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                posture_state = "Standing" if avg_shoulder_y < 0.4 else "Sitting"

                # Activity Logic
                avg_wrist_x = (left_wrist.x + right_wrist.x) / 2
                avg_wrist_y = (left_wrist.y + right_wrist.y) / 2
                wrist_history.append((avg_wrist_x, avg_wrist_y))

                if len(wrist_history) == fps:
                    start_x, start_y = wrist_history[0]
                    end_x, end_y = wrist_history[-1]
                    movement_distance = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
                    activity_state = "Working" if movement_distance > 0.02 else "Idle"

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )

            # Draw Dashboard Overlay
            cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"Presence: {presence_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Posture:  {posture_state}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"Activity: {activity_state}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if activity_state == "Working" else (0, 0, 255),
                2,
            )

            writer.writerow([frame_count, timestamp, presence_state, posture_state, activity_state])
            out.write(frame)

            cv2.imshow("Employee Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete! Saved video to '{output_video_path}' and report to '{csv_report_path}'.")


if __name__ == "__main__":
    main()
