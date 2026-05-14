import numpy as np


class EmotionClassifier:
    def __init__(self, backend="auto", confidence_threshold=0.4):
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._init_backend()

    def _init_backend(self):
        if self.backend in ("auto", "fer"):
            try:
                from fer.fer import FER

                self.model = FER(mtcnn=False)
                self.backend = "fer"
                return
            except Exception:
                if self.backend == "fer":
                    raise

        if self.backend in ("auto", "deepface"):
            try:
                from deepface import DeepFace

                self.model = DeepFace
                self.backend = "deepface"
                return
            except Exception:
                if self.backend == "deepface":
                    raise

        raise RuntimeError("No emotion backend available. Install 'fer' or 'deepface'.")

    def predict_emotions(self, frame_bgr, faces):
        results = []
        for (x, y, w, h) in faces:
            face_bgr = frame_bgr[y : y + h, x : x + w]
            if face_bgr.size == 0:
                continue
            if self.backend == "fer":
                label, score = self._predict_fer(face_bgr)
                emotions = {label: score} if label else {}
            else:
                label, score, emotions = self._predict_deepface(face_bgr)

            if not label or score < self.confidence_threshold:
                label = "Uncertain"

            results.append(
                {
                    "box": (x, y, w, h),
                    "label": label,
                    "score": score,
                    "emotions": emotions,
                }
            )
        return results

    def _predict_fer(self, face_bgr):
        try:
            import cv2

            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            label, score = self.model.top_emotion(face_rgb)
            if label is None:
                return None, 0.0
            return label.title(), float(score)
        except Exception:
            return None, 0.0

    def _predict_deepface(self, face_bgr):
        try:
            import cv2

            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            result = self.model.analyze(
                face_rgb,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            emotions = result.get("emotion", {})
            if not emotions:
                return None, 0.0, {}
            label = max(emotions, key=emotions.get)
            score = float(emotions[label]) / 100.0
            return label.title(), score, emotions
        except Exception:
            return None, 0.0, {}
