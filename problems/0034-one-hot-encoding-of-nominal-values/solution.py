import numpy as np

def to_categorical(y, num_classes=None):

    y = np.array(y, dtype='int')
    
    if num_classes is None:
        num_classes = np.max(y) + 1
        
    n = y.shape[0]
    
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    
    categorical[np.arange(n), y] = 1.0
    
    return categorical