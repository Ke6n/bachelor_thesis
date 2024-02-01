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
    percentage_error = 100*(y_true - y_pred)/y_true
    return np.abs(percentage_error)

def squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    __input_check(y_true, y_pred)
    percentage_error = 100*(y_true - y_pred)/y_true
    return np.square(percentage_error)

def symmetric_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    __input_check(y_true, y_pred)
    diff = y_true - y_pred
    sum = y_true + y_pred
    return 200*np.abs(diff/sum)
