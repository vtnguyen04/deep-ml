import numpy as np

def stratified_kfold_indices(y, n_splits):
	"""
	Generate train/test indices for stratified K-fold cross-validation.

	Args:
		y: 1D array-like of integer class labels
		n_splits: number of folds

	Returns:
		A list of [train_indices, test_indices] pairs, one per fold.
	"""
	y = np.array(y)
	unique_classes = np.unique(y)

	class_splits = [np.array_split(np.where(y == cls)[0], n_splits) for cls in unique_classes]

	folds = []
	n_samples = len(y)

	indices = np.arange(n_samples)

	for fold_idx in range(n_splits):

		test_idx = []
		for splits in class_splits:
			test_idx.extend(splits[fold_idx])
		test_idx.sort() 

		mask = np.ones(n_samples, dtype=bool)
		mask[test_idx] = False
		train_idx = indices[mask]

		folds.append([train_idx.tolist(), test_idx])

	return folds

	
	