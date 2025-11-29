import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# === Load embeddings from HDF5 ===
file_path = "V2_250_multi_embeddings.h5"
layer_keys = []

with h5py.File(file_path, mode='r') as ifh:

    print(f'Here are the datasets present in the h5 file {file_path}:')
    for item in ifh.keys():
        print('\t', item)
        if item != 'Labels':
            layer_keys.append(item)

    if 'Labels' in ifh.keys():
            all_labels = np.array(ifh['Labels'])
            print('Class distribution:', {int(k): int(v) for k, v in zip(*np.unique(all_labels, return_counts=True))})
    else:
        raise ValueError('No Labels present in the h5 file')

    print(f'There were a total of {len(layer_keys)} in this h5 file')

    for layer in layer_keys:

        X = np.array(ifh[layer])
        labels = all_labels[:len(X)]
        print(f'Loaded {len(X)} embeddings from layer: {layer}')
        print(f'Splitting and training layer {layer}')
        # === Train/test split and classification as before ===
        X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", C=2.0))

        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_val)[:, 1]
        y_pred05 = (probs >= 0.5).astype(int)

        print('\n\t=== Default threshold (0.5) ===')
        print('\tAccuracy:', accuracy_score(y_val, y_pred05))
        print('\tPred counts:', {int(k): int(v) for k, v in zip(*np.unique(y_pred05, return_counts=True))})
        print('\t',             classification_report(y_val, y_pred05, digits=4, zero_division=0))
        try:
            print("ROC-AUC:", roc_auc_score(y_val, probs))
        except ValueError:
            print("ROC-AUC: cannot compute (only one class present)")

        print("Confusion matrix:\n", confusion_matrix(y_val, y_pred05))