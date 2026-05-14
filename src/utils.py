import time


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class FpsCounter:
    def __init__(self, avg_window=10):
        self.avg_window = avg_window
        self.timestamps = []

    def tick(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.avg_window:
            self.timestamps.pop(0)

    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / duration
