"""Threshold-based trading strategy."""


class ThresholdStrategy:
    def __init__(self, threshold: float = 0.55, long_only: bool = True):
        self.threshold = threshold
        self.long_only = long_only

    def signal(self, probability: float) -> int:
        if self.long_only:
            return 1 if probability > self.threshold else 0
        else:
            if probability > self.threshold:
                return 1
            elif probability < (1 - self.threshold):
                return -1
            else:
                return 0
