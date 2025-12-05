import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    precision_recall_curve)

import matplotlib.pyplot as plt
import seaborn as sns

## Perceptron specific imports ##
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# === Load embeddings from HDF5 ===
file_path = 'V2_500_multi_embeddings_expanded_CLS_separate_N_labeled_seqs.h5'
THRESHOLD = 0.5

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

    print(f'There were a total of {int(len(layer_keys)/2)} layers in this h5 file')
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", C=2.0))

    clf_mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=1e-4,  # L2 regularization
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=200,  # or more
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
            )
        )
    for layer_index in range(10, 11):
    #for layer_index in range(int(len(layer_keys)/2)):

        X_layer = np.array(ifh['Layer' + str(layer_index)])
        X_cls = np.array(ifh['CLS' + str(layer_index)])
        X = np.concatenate([X_cls, X_layer], axis=1)

        labels = all_labels[:len(X)]
        print(f'\nLoaded {len(X)} embeddings from layer: {layer_index}')
        print(f'Splitting and training layer {layer_index}')
        # === Train/test split and classification as before ===
        X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

        # print('\nCalculating AUC cross-validations for logistic regression')
        # cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        # scores = cross_val_score(clf, X, labels, cv=cv, scoring='roc_auc')
        # print('\nMean AUC over folds from LR:', scores.mean())
        # print('AUC variance over folds from LR:', scores.var())
        # print('LR raw scores', scores)
        #
        # print('\nCalculating AUC cross-validations for multilayer perceptron')
        # cv_mlp = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        # scores_mlp = cross_val_score(clf_mlp, X, labels, cv=cv_mlp, scoring='roc_auc')
        # print('Mean AUC over folds from MLP:', scores_mlp.mean())
        # print('AUC variance over folds from MLP:', scores_mlp.var())
        # print('MLP raw scores', scores_mlp)

        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)[:, 1]
        y_pred = (probs >= THRESHOLD).astype(int)

        print(f'\n\t=== Threshold set at {THRESHOLD} ===')
        print('\tAccuracy:', accuracy_score(y_val, y_pred))
        print('\tPred counts:', {int(k): int(v) for k, v in zip(*np.unique(y_pred, return_counts=True))})
        print('\t',             classification_report(y_val, y_pred, digits=4, zero_division=0))
        try:
            print("ROC-AUC:", roc_auc_score(y_val, probs))
        except ValueError:
            print("ROC-AUC: cannot compute (only one class present)")

    # ----------------------------------------------------
    # Confusion matrix
    # ----------------------------------------------------
        cm = confusion_matrix(y_val, y_pred)
        cm_labels = ["Drug Sensitive", "Drug Resistant"]
        print("Confusion matrix:\n", cm)

    # ----------------------------------------------------
    # Precision-Recall
    # ----------------------------------------------------
    prec, rec, pr_thresh = precision_recall_curve(y_val, probs)
    pr_auc = auc(rec, prec)

    # ----------------------------------------------------
    # ROC curve
    # ----------------------------------------------------
    fpr, tpr, roc_thresh = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)

    # ----------------------------------------------------
    # Threshold sweep for accuracy, FP, FN
    # ----------------------------------------------------
    thresholds = np.linspace(0, 1, 200)
    accuracies = []
    false_pos = []
    false_neg = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        cm_t = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm_t.ravel()
        accuracies.append((tp + tn) / (tp + tn + fp + fn))
        false_pos.append(fp)
        false_neg.append(fn)

    accuracies = np.array(accuracies)
    false_pos = np.array(false_pos)
    false_neg = np.array(false_neg)

    # ----------------------------------------------------
    # Plotting (extended: 6-panel professional layout)
    # ----------------------------------------------------
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.25)

    # ============================
    # Panel 1: Confusion matrix
    # ============================
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=cm_labels,
        yticklabels=cm_labels,
        cbar=False,
        square=True,
        annot_kws={"size": 14},
        ax=ax1
    )
    ax1.set_title(f"Confusion Matrix (Threshold = {THRESHOLD})", fontsize=16)
    ax1.set_xlabel("Predicted Label", fontsize=12)
    ax1.set_ylabel("True Label", fontsize=12)

    # ============================
    # Panel 2: Precision–Recall (full)
    # ============================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rec, prec, linewidth=2)
    ax2.set_title(f"Precision–Recall Curve (AUC = {pr_auc:.3f})", fontsize=16)
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # draw vertical line for chosen THRESHOLD if we can map it to a recall
    if len(pr_thresh) > 0:
        # find threshold index closest to THRESHOLD
        idx = np.argmin(np.abs(pr_thresh - THRESHOLD))
        # pr_thresh aligns with precision/recall points; precision_recall_curve returns
        # thresholds with length = len(prec)-1; map to recall[idx]
        rec_at_thr = rec[idx]
        prec_at_thr = prec[idx]
        ax2.axvline(rec_at_thr, linestyle="--", color="tab:gray", linewidth=1.5)
        # Mark the point on the curve
        ax2.plot(rec_at_thr, prec_at_thr, marker='o', markersize=6, color='tab:gray',
                 label=f"thr={THRESHOLD:.2f}")
        ax2.legend(loc='lower left', fontsize=10)

    # ============================
    # Panel 3: ROC Curve
    # ============================
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(fpr, tpr, linewidth=2)
    ax3.plot([0, 1], [0, 1], linestyle="--", color="grey")

    idx = np.argmin(np.abs(roc_thresh - THRESHOLD))
    tpr_at_thr = tpr[idx]
    fpr_at_thr = fpr[idx]

    # Vertical line at this FPR
    ax3.axvline(fpr_at_thr, linestyle="--", color="tab:gray", linewidth=1.5)
    # Mark the point on ROC
    ax3.plot(fpr_at_thr, tpr_at_thr, 'o', color="tab:gray", markersize=6,
             label=f"thr={THRESHOLD:.2f}")

    ax3.legend(loc="lower right")

    ax3.set_title(f"ROC Curve (AUC = {roc_auc:.3f})", fontsize=16)
    ax3.set_xlabel("False Positive Rate", fontsize=12)
    ax3.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # ============================
    # Panel 4: Threshold curves (full)
    # ============================
    ax4 = fig.add_subplot(gs[2, 0])

    # Accuracy (left axis)
    ax4.plot(thresholds, accuracies, linewidth=2, label="Accuracy")
    ax4.set_xlabel("Decision Threshold", fontsize=12)
    ax4.set_ylabel("Accuracy", fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.grid(True, linestyle="--", alpha=0.4)

    # FP and FN counts (right axis)
    ax4b = ax4.twinx()
    ax4b.plot(thresholds, false_pos, linestyle="--", linewidth=2, label="False Positives", color="tab:red")
    ax4b.plot(thresholds, false_neg, linestyle="--", linewidth=2, label="False Negatives", color="tab:green")
    ax4b.set_ylabel("Counts", fontsize=12)

    # vertical line at chosen threshold
    ax4.axvline(THRESHOLD, linestyle="--", color="tab:gray", linewidth=1.5)

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="center right")

    ax4.set_title("Threshold Sweep: Accuracy, FP, FN", fontsize=16)

    # ============================
    # Panel 5: Zoomed PR Curve (Recall 0.8–1.0)
    # ============================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(rec, prec, linewidth=2)
    ax5.set_xlim(0.8, 1.0)
    ax5.set_ylim(bottom=0.4)  # let top auto-scale
    ax5.set_title("Zoomed PR Curve (Recall 0.8–1.0)", fontsize=14)
    ax5.set_xlabel("Recall", fontsize=12)
    ax5.set_ylabel("Precision", fontsize=12)
    ax5.grid(True, linestyle="--", alpha=0.4)

    # mark threshold on zoomed PR if mapping exists
    if len(pr_thresh) > 0:
        ax5.axvline(rec_at_thr, linestyle="--", color="tab:gray", linewidth=1.5)
        ax5.plot(rec_at_thr, prec_at_thr, marker='o', markersize=6, color='tab:gray')

    # ============================
    # Panel 6: Zoomed Threshold Sweep (0.3–0.8)
    # ============================
    ax6 = fig.add_subplot(gs[2, 1])

    mask = (thresholds >= 0.3) & (thresholds <= 0.8)
    ax6.plot(thresholds[mask], accuracies[mask], linewidth=2, label="Accuracy")
    ax6.set_xlabel("Decision Threshold", fontsize=12)
    ax6.set_ylabel("Accuracy", fontsize=12)
    ax6.set_ylim(0.85, 1)
    ax6.grid(True, linestyle="--", alpha=0.4)

    ax6b = ax6.twinx()
    ax6b.plot(thresholds[mask], false_pos[mask], linestyle="--", linewidth=2, label="False Positives", color="tab:red")
    ax6b.plot(thresholds[mask], false_neg[mask], linestyle="--", linewidth=2, label="False Negatives",
              color="tab:green")
    ax6b.set_ylabel("Counts", fontsize=12)

    # vertical line for chosen threshold if in zoom range
    if 0.3 <= THRESHOLD <= 0.8:
        ax6.axvline(THRESHOLD, linestyle="--", color="tab:gray", linewidth=1.5)

    # Combined legend
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper right")

    ax6.set_title("Zoomed Threshold Sweep (0.3–0.8)", fontsize=14)

    plt.tight_layout()

    plt.show()