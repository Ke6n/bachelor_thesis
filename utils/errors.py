#!/usr/bin/env python3
import numpy as np

def __input_check(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.ndim != y_pred.ndim:
        raise ValueError("Equal dimension required for y_true and y_pred")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Dimension of y_true and y_pred must be 1")

def absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    __input_check(y_true, y_pred)
    diff = y_true - y_pred
    return np.abs(diff)

def squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    __input_check(y_true, y_pred)
    diff = y_true - y_pred
    return np.square(diff)

def absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    __input_check(y_true, y_pred)
    percentage_error = np.true_divide(100*(y_true - y_pred), y_true)
    return np.abs(percentage_error)

def squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    __input_check(y_true, y_pred)
    percentage_error = np.true_divide(100*(y_true - y_pred), y_true)
    return np.square(percentage_error)

def symmetric_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    __input_check(y_true, y_pred)
    diff = np.abs(y_true - y_pred)
    sum = np.abs(y_true) + np.abs(y_pred)
    return 2*np.true_divide(diff, sum)

def modified_symmetric_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    __input_check(y_true, y_pred)
    diff = np.abs(y_true - y_pred)
    n = len(y_true)
    cumsum_y = np.cumsum(y_true)
    cumsum_y = cumsum_y[:n-1]
    cumsum_divisor = np.linspace(1, n-1, n-1)
    mean_cumsum_y = np.true_divide(cumsum_y, cumsum_divisor)
    s = np.zeros(n)
    for i in range(1,n):
        pref_y = y_true[:i]
        sum_abs = np.sum(np.abs(pref_y - mean_cumsum_y[i-1]))
        s[i] = np.true_divide(sum_abs, i)
    sum = np.abs(y_true) + np.abs(y_pred)
    divisor = np.true_divide(sum, 2) + s
    return np.true_divide(diff, divisor)

def relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray) -> np.ndarray:
    diff = y_true - y_pred
    diff_bm = y_true - bm_pred
    return np.abs(np.true_divide(diff, diff_bm))

def scaled_error(y_true: np.ndarray, y_pred: np.ndarray, y_in_sample: np.ndarray) -> np.ndarray:
    diff_in_sample = np.delete(y_in_sample, 0) - np.delete(y_in_sample, -1)
    diff = y_true - y_pred
    return np.true_divide(diff, np.mean(np.abs(diff_in_sample)))

def bounded_RAE(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray) -> np.ndarray:
    diff = np.abs(y_true - y_pred)
    diff_bm = np.abs(y_true - bm_pred)
    return np.true_divide(diff, (diff + diff_bm))