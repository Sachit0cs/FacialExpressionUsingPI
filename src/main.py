import cv2
from .capture import open_camera
from .face_detector import FaceDetector
from .emotion_classifier import EmotionClassifier
from .ui import draw_overlays, compose_ui
from .utils import FpsCounter


def choose_mood(detections):
    if not detections:
        return "Neutral"

    # Pick highest-confidence detection
    best = max(detections, key=lambda d: d["score"])
    return best["label"]


def run_app():
    camera_index = 0
    cap = open_camera(camera_index, width=640, height=480)
    if cap is None:
        print("Error: Could not open webcam. Try a different camera index.")
        return

    detector = FaceDetector()
    classifier = EmotionClassifier(backend="auto", confidence_threshold=0.4)
    fps_counter = FpsCounter(avg_window=15)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_counter.tick()
        faces = detector.detect_faces(frame)
        detections = classifier.predict_emotions(frame, faces)
        mood_label = choose_mood(detections)

        draw_overlays(frame, detections, fps_counter.get_fps(), mood_label)
        ui_frame = compose_ui(frame, mood_label)

        cv2.imshow("Emotion Detection Mood Lamp", ui_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_app()
