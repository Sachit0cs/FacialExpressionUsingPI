import cv2


def open_camera(camera_index=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
