import cv2
import numpy as np
from .mood_lamp import emotion_to_color, create_lamp_panel


def draw_overlays(frame, detections, fps, mood_label):
    for det in detections:
        x, y, w, h = det["box"]
        label = det["label"]
        score = det["score"]
        color = emotion_to_color(label)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {score * 100:.0f}%"
        cv2.putText(
            frame,
            text,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    mood_text = f"Mood State: {mood_label}"
    cv2.putText(
        frame,
        mood_text,
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def compose_ui(frame, mood_label):
    color = emotion_to_color(mood_label)
    lamp_panel = create_lamp_panel(260, frame.shape[0], color)

    # Background tint
    tint = np.full_like(frame, color)
    blended = cv2.addWeighted(frame, 0.8, tint, 0.2, 0)

    return np.hstack([blended, lamp_panel])
