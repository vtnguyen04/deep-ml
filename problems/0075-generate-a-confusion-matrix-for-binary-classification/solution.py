
from collections import Counter
import numpy as np
def confusion_matrix(data):
  arr = np.asarray(data)
  y_true, y_pred = arr[:, 0], arr[:, 1]

  tp = np.sum((y_true == 1) & (y_pred == 1))
  fn = np.sum((y_true == 1) & (y_pred == 0))
  fp = np.sum((y_true == 0) & (y_pred == 1))
  tn = np.sum((y_true == 0) & (y_pred == 0))

  return [[int(tp), int(fn)], [int(fp), int(tn)]]