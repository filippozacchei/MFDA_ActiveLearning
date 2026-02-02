import numpy as np


class CoarseOutput(np.ndarray):
    """
    NumPy-compatible array that also stores prediction variance.
    """

    def __new__(cls, y_pred: np.ndarray, y_var: np.ndarray):
        obj = np.asarray(y_pred).view(cls)
        obj.variance = y_var
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.variance = getattr(obj, "variance", None)
