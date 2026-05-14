import numpy as np
import cv2


EMOTION_COLORS = {
    "Happy": (0, 255, 0),
    "Angry": (0, 0, 255),
    "Sad": (255, 0, 0),
    "Neutral": (255, 255, 255),
    "Surprise": (0, 255, 255),
    "Uncertain": (128, 128, 128),
}


def emotion_to_color(label):
    return EMOTION_COLORS.get(label, (128, 128, 128))


def create_lamp_panel(width, height, color_bgr, intensity=0.85):
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    # Soft gradient background
    for i in range(height):
        blend = i / max(1, height - 1)
        panel[i, :] = (color_bgr[0] * blend, color_bgr[1] * blend, color_bgr[2] * blend)

    # Glow circle
    center = (width // 2, height // 2)
    max_radius = min(width, height) // 3
    for r in range(max_radius, 0, -1):
        alpha = (r / max_radius) ** 2
        glow = (
            int(color_bgr[0] * alpha * intensity),
            int(color_bgr[1] * alpha * intensity),
            int(color_bgr[2] * alpha * intensity),
        )
        cv2.circle(panel, center, r, glow, -1, lineType=cv2.LINE_AA)

    # Lamp base
    base_top = (center[0] - max_radius // 2, center[1] + max_radius // 2)
    base_bottom = (center[0] + max_radius // 2, center[1] + max_radius)
    cv2.rectangle(panel, base_top, base_bottom, (30, 30, 30), -1)

    return panel
