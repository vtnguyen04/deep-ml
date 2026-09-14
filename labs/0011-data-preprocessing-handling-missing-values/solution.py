import numpy as np


def impute(X: np.ndarray) -> np.ndarray:
  """Fill in missing values (NaN) in the input array.

  Args:
      X: Array with possible NaN values, shape (n_samples, n_features)

  Returns:
      X_clean: Array with no NaN values, same shape as X
  """
  X_clean = np.array(X, dtype=float, copy=True)
  nan_mask = np.isnan(X_clean)

  if not np.any(nan_mask):
    return X_clean

  n_samples, n_features = X_clean.shape

  col_medians = np.zeros(n_features)
  col_mins = np.zeros(n_features)
  col_maxs = np.zeros(n_features)

  for j in range(n_features):
    valid_vals = X_clean[~nan_mask[:, j], j]
    if len(valid_vals) > 0:
      col_medians[j] = np.median(valid_vals)
      col_mins[j] = np.min(valid_vals)
      col_maxs[j] = np.max(valid_vals)
    else:
      col_medians[j] = 0.0
      col_mins[j] = 0.0
      col_maxs[j] = 0.0

    X_clean[nan_mask[:, j], j] = col_medians[j]

  cols_with_nan = [j for j in range(n_features) if np.any(nan_mask[:, j])]
  alpha = 1e-2  
  for _ in range(5):
    for j in cols_with_nan:
      obs_mask = ~nan_mask[:, j]
      miss_mask = nan_mask[:, j]

      if np.sum(obs_mask) < 2:
        continue

      predictor_cols = [c for c in range(n_features) if c != j]

      n_obs = np.sum(obs_mask)
      n_miss = np.sum(miss_mask)

      X_obs = np.hstack(
          [np.ones((n_obs, 1)), X_clean[obs_mask][:, predictor_cols]]
      )
      y_obs = X_clean[obs_mask, j]

      X_miss = np.hstack(
          [np.ones((n_miss, 1)), X_clean[miss_mask][:, predictor_cols]]
      )


      d = X_obs.shape[1]
      A = X_obs.T @ X_obs + alpha * np.eye(d)
      b = X_obs.T @ y_obs

      try:
        w = np.linalg.solve(A, b)
        preds = X_miss @ w

        preds = np.clip(preds, col_mins[j], col_maxs[j])
        X_clean[miss_mask, j] = preds
      except np.linalg.LinAlgError:
        pass

  np.nan_to_num(X_clean, copy=False, nan=0.0)
  return X_clean