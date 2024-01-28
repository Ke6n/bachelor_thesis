#!/usr/bin/env python3
import numpy as np
import errors

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    abs_error = errors.absolute_error(y_true, y_pred)
    return np.mean(abs_error)

mae = mean_absolute_error

