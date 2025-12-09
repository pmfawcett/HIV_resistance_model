import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    cross_validate
)
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
    precision_recall_curve
)
from textwrap import fill

# ---------------------------------------
# Alternative classifier-specific imports
#----------------------------------------

from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

EMBEDDINGS_FILE_PATH = 'V2_500_multi_embeddings_expanded_CLS_separate_N_labeled_seqs.h5'
THRESHOLD = 0.5
LAYERS_TO_ANALYZE = (10,11)  # list here as tuple or range. Add a trailing comma if only one item
# For all layers, set to range(int(len(layer_keys)/2)):

# -----------------------------------------------------------------------------
# Make pipelines for Logistic regression and optional alternative classifiers
# -----------------------------------------------------------------------------

clf = make_pipeline(
    StandardScaler(),  # This makes the pipeline for logistic regression for use later
    LogisticRegression(
        max_iter=3000, 
        class_weight='balanced',
        solver='lbfgs', 
        C=2.0)
)

clf_mlp = make_pipeline(  # This is only if you are using a multilayer perceptron later
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

clf_LGBM = make_pipeline(  # This is only if you are using the LightGBM classifier
    LGBMClassifier(
        class_weight='balanced',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary'
    )
)


AUC_cat_results = {}
AUC_seq_results = {}
recall_seq_results = {}
recall_cat_results = {}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # Cross-validation split for use later

# ------------------------------------
# Open HD5 file and determine datasets
# ------------------------------------

with (h5py.File(EMBEDDINGS_FILE_PATH, mode='r') as ifh):
    print(f'Datasets present in the h5 file {EMBEDDINGS_FILE_PATH}:')
    print(fill(', '.join(ifh.keys()), width=120))
    layer_keys = [k for k in ifh.keys() if k != 'Labels']
    if 'Labels' in ifh.keys():
        all_labels = np.array(ifh['Labels'])
        print('\nClass distribution:', {int(k): int(v) for k, v in zip(*np.unique(all_labels, return_counts=True))})
    else:
        raise ValueError('No data labels present in the h5 file')

    print(f'There were a total of {int(len(layer_keys) / 2)} layers in this h5 file')

    # ---------------------------------------------------------------------------------------
    # Main loop over layers user wishes to analyze. Train and report on data from each layer.
    # ---------------------------------------------------------------------------------------

    for layer_index in LAYERS_TO_ANALYZE :

        X_seq = np.array(ifh['Layer' + str(layer_index)])
        X_cls = np.array(ifh['CLS' + str(layer_index)])
        X_cat = np.concatenate([X_cls, X_seq], axis=1)  # Concatenate CLS and sequence embedding tokens

        labels = all_labels[:len(X_cat)]
        print(f'\nSplitting and training {len(X_cat)} embeddings from layer {layer_index}:')
        # === Train/test split and classification as before ===
        X_cat_train, X_cat_val, _, _ = train_test_split(
     X_cat,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        X_seq_train, X_seq_val, _, _ = train_test_split(
            X_seq,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        X_cls_train, X_cls_val, y_train, y_val = train_test_split(
            X_cls,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        
        print('\nCalculating CLS+SEQ cross-validations for logistic regression:')
        scores = cross_validate(clf, X_cat, labels, cv=10, scoring=['recall', 'roc_auc'])
        print('\tMean CLS+SEQ AUC over folds from LR:', scores['test_roc_auc'].mean())
        print('\tCLS+SEQ AUC variance over folds from LR:', scores['test_roc_auc'].var())
        # print('\tCLS+SEQ LR raw AUC scores', scores['test_roc_auc'])
        print('\tMean CLS+SEQ recall over folds from LR:', scores['test_recall'].mean())
        print('\tCLS+SEQ recall variance over folds from LR:', scores['test_recall'].var())
        # print('\tCLS+SEQ LR raw recall scores', scores['test_recall'])
        AUC_cat_results['Layer '+str(layer_index)] = scores['test_roc_auc'].mean()
        recall_cat_results['Layer ' + str(layer_index)] = scores['test_recall'].mean()

        # print('\nCalculating SEQ cross-validations for logistic regression:')
        # scores = cross_val_score(clf, X_seq, labels, cv=cv, scoring='roc_auc')
        # print('Mean SEQ AUC over folds from LR:', scores.mean())
        # print('SEQ AUC variance over folds from LR:', scores.var())
        # print('SEQ LR raw AUC scores', scores)
         # AUC_seq_results['Layer ' + str(layer_index)] = scores.mean()

        print('\nCalculating SEQ cross-validations for logistic regression:')
        scores = cross_validate(clf, X_seq, labels, cv=10, scoring=['recall', 'roc_auc'])
        print('\tMean SEQ AUC over folds from LR:', scores['test_roc_auc'].mean())
        print('\tSEQ AUC variance over folds from LR:', scores['test_roc_auc'].var())
        # print('\tSEQ LR raw AUC scores', scores['test_roc_auc'])
        print('\tMean SEQ recall over folds from LR:', scores['test_recall'].mean())
        print('\tSEQ recall variance over folds from LR:', scores['test_recall'].var())
        # print('\tSEQ LR raw recall scores', scores['test_recall'])

        AUC_seq_results['Layer ' + str(layer_index)] = scores['test_roc_auc'].mean()
        recall_seq_results['Layer ' + str(layer_index)] = scores['test_recall'].mean()

        # print('\nCalculating CLS cross-validations for logistic regression:')
        # scores = cross_val_score(clf, X_seq, labels, cv=cv, scoring='roc_auc')
        # print('Mean CLS AUC over folds from LR:', scores.mean())
        # print('CLS AUC variance over folds from LR:', scores.var())
        # print('CLS LR raw AUC scores', scores)

        #
        # print('\nCalculating recall cross-validations for logistic regression:')
        # scores = cross_val_score(clf, X, labels, cv=cv, scoring='recall')
        # print('Mean recall over folds from LR:', scores.mean())
        # print('Recall variance over folds from LR:', scores.var())
        # print('LR raw recall scores', scores)

        # print('\nCalculating AUC cross-validations for multilayer perceptron')
        # scores_mlp = cross_val_score(clf_mlp, X, labels, cv=cv, scoring='roc_auc')
        # print('Mean AUC over folds from MLP:', scores_mlp.mean())
        # print('AUC variance over folds from MLP:', scores_mlp.var())
        # print('MLP raw scores', scores_mlp)
        #
        # print('\nCalculating recall cross-validations for LightGBM:')
        # scores = cross_val_score(clf_LGBM, X, labels, cv=cv, scoring='recall')
        # print('Mean recall over folds from LGBM:', scores.mean())
        # print('Recall variance over folds from LGBM:', scores.var())
        # print('LGBM raw recall scores', scores)

        print('\nFitting model using concatenated CLS and mean-pooled sequence tokens')
        clf.fit(X_cat_train, y_train)
        probs_cat: np.ndarray = clf.predict_proba(X_cat_val)[:, 1]
        y_cat_pred = (probs_cat >= THRESHOLD).astype(int)

        print('Fitting model using CLS tokens alone')
        clf.fit(X_cls_train, y_train)
        probs_cls: np.ndarray = clf.predict_proba(X_cls_val)[:, 1]
        y_cls_pred = (probs_cls >= THRESHOLD).astype(int)

        print('Fitting model using mean-pooled sequence tokens alone')
        clf.fit(X_seq_train, y_train)
        probs_seq: np.ndarray = clf.predict_proba(X_seq_val)[:, 1]
        y_seq_pred = (probs_seq >= THRESHOLD).astype(int)

        # ----------------------------------------------------
        # ROC curve calculations
        # ----------------------------------------------------

        fpr_cls, tpr_cls, roc_thresh_cls = roc_curve(y_val, probs_cls)
        roc_auc_cls = auc(fpr_cls, tpr_cls)

        fpr_seq, tpr_seq, roc_thresh_seq = roc_curve(y_val, probs_seq)
        roc_auc_seq = auc(fpr_seq, tpr_seq)

        fpr_cat, tpr_cat, roc_thresh_cat = roc_curve(y_val, probs_cat)
        roc_auc_cat = auc(fpr_cat, tpr_cat)

        # ------------------------------------------------------------------
        # Decide which set of values to report  (choose _seq, _cls, or _cat)
        # ------------------------------------------------------------------

        print('\nPlots and metric data are based on mean-pooled SEQ tokens')

        y_pred, y_val, probs = y_seq_pred, y_val, probs_seq
        fpr, tpr, roc_thresh, roc_auc = fpr_seq, tpr_seq, roc_thresh_seq, roc_auc_seq

        # ----------------------------------
        # Reports including confusion matrix
        # ----------------------------------

        print(f'\n\t=== Classifier threshold set at {THRESHOLD} ===')
        print('\tAccuracy:', accuracy_score(y_val, y_pred))
        print('\tROC-AUC:', roc_auc_score(y_val, probs))
        print('\tPred counts:', {int(k): int(v) for k, v in zip(*np.unique(y_pred, return_counts=True))})
        print('\t', classification_report(y_val, y_pred, digits=4, zero_division=0))

        cm = confusion_matrix(y_val, y_cat_pred)
        print('\nCLS+SEQ token confusion matrix:\n', cm)
        cm = confusion_matrix(y_val, y_seq_pred)
        print('\nSEQ token alone confusion matrix:\n', cm)
        cm_labels = ['Drug Sensitive', 'Drug Resistant']


# ------------------------------------------------------------
# Decide on which token strategy worked best for chosen layers
# ------------------------------------------------------------

print(f'\nAnalysing winning token strategy for {", ".join(AUC_seq_results)}:\n')
for k in AUC_seq_results:
    print(f'\tFor {k}:')
    print(f'\t\tSEQ results for {k}: recall mean {recall_seq_results[k]}, AUC mean {AUC_seq_results[k]}')
    print(f'\t\tCLS+SEQ results for {k}: recall mean {recall_cat_results[k]}, AUC mean {AUC_cat_results[k]}')
    if AUC_seq_results[k] > AUC_cat_results[k]:
        print(f'\t\tSEQ alone performed best for ROC-AUC on {k}')
    else:
        print(f'\t\tCLS+SEQ concatenated performed best for ROC-AUC on {k}')

k_cat, v_cat = max(AUC_cat_results.items(), key=lambda x: x[1])
k_seq, v_seq = max(AUC_seq_results.items(), key=lambda x: x[1])

print(f'\nChampion ROC-AUC layer for CLS+SEQ was {k_cat} with mean AUC: {v_cat}')
print(f'Champion ROC-AUC layer for SEQ was {k_seq} with mean AUC: {v_seq}')

k_cat, v_cat = max(recall_cat_results.items(), key=lambda x: x[1])
k_seq, v_seq = max(recall_seq_results.items(), key=lambda x: x[1])

print(f'\nChampion recall layer for CLS+SEQ was {k_cat} with mean AUC: {v_cat}')
print(f'Champion recall layer for SEQ was {k_seq} with mean AUC: {v_seq}')

# ----------------
# Precision-Recall
# ----------------

prec_cat, rec_cat, pr_thresh_cat = precision_recall_curve(y_val, probs_cat)
prec, rec, pr_thresh = precision_recall_curve(y_val, probs)
pr_auc = auc(rec, prec)
prec_cls, rec_cls, pr_thresh_cls = precision_recall_curve(y_val, probs_cls)
prec_seq, rec_seq, pr_thresh_seq = precision_recall_curve(y_val, probs_seq)

# ------------------------------------
# Threshold sweep for accuracy, FP, FN
# ------------------------------------

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

# -------------------------
# Plotting (6-panel layout)
# -------------------------
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.25)

# -------------------------
# Panel 1: Confusion matrix
# -------------------------
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(
    cm,
    annot=True, fmt='d', cmap='Blues',
    xticklabels=cm_labels,
    yticklabels=cm_labels,
    cbar=False,
    square=True,
    annot_kws={'size': 14},
    ax=ax1
)
ax1.set_title(f'Confusion Matrix (Threshold = {THRESHOLD})', fontsize=16)
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_ylabel('True Label', fontsize=12)

# ------------------
# Panel 2: ROC curve
# ------------------

ax3 = fig.add_subplot(gs[0, 1])
ax3.plot(fpr_cat, tpr_cat, linewidth=1, color='tab:blue', label='CLS+SEQ')
ax3.plot(fpr_seq, tpr_seq, linewidth=1, color='tab:green', label='SEQ')
ax3.plot(fpr_cls, tpr_cls, linewidth=1, color='tab:red', label='CLS')
ax3.plot([0, 1], [0, 1], linestyle='--', color='grey')

idx = np.argmin(np.abs(roc_thresh - THRESHOLD))
tpr_at_thr = tpr[idx]
fpr_at_thr = fpr[idx]

# Vertical line at this FPR
ax3.axvline(fpr_at_thr, linestyle='--', color='tab:gray', linewidth=1.5)
# Mark the point on ROC
ax3.plot(fpr_at_thr, tpr_at_thr, 'o', color='tab:gray', markersize=6,
         label=f'thr={THRESHOLD:.2f}')

ax3.legend(loc='lower right')

ax3.set_title(f'ROC Curve (AUC = {roc_auc:.3f})', fontsize=16)
ax3.set_xlabel('False Positive Rate', fontsize=12)
ax3.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.4)

# --------------------------------
# Panel 3: Precision–Recall (full)
# --------------------------------
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(rec_cat, prec_cat, linewidth=1, color='tab:blue', label='CLS+SEQ')
ax2.plot(rec_cls, prec_cls, linewidth=1, color='tab:red', label='CLS')
ax2.plot(rec_seq, prec_seq, linewidth=1, color='tab:green', label='SEQ')
ax2.set_title(label=f'Precision–Recall Curve (AUC = {pr_auc:.3f})', fontsize=16)
ax2.set_xlabel(xlabel='Recall', fontsize=12)
ax2.set_ylabel(ylabel='Precision', fontsize=12)
ax2.grid(visible=True, linestyle='--', alpha=0.4)

# draw vertical line for chosen THRESHOLD if we can map it to a recall
if len(pr_thresh) > 0:
    # find threshold index closest to THRESHOLD
    idx = np.argmin(np.abs(pr_thresh - THRESHOLD))
    # pr_thresh aligns with precision/recall points; precision_recall_curve returns
    # thresholds with length = len(prec)-1; map to recall[idx]
    rec_at_thr = rec[idx]
    prec_at_thr = prec[idx]
    ax2.axvline(rec_at_thr, linestyle='--', color='tab:gray', linewidth=1.5)
    # Mark the point on the curve
    ax2.plot(rec_at_thr, prec_at_thr, marker='o', markersize=6, color='tab:gray',
             label=f'thr={THRESHOLD:.2f}')
    ax2.legend(loc='lower left', fontsize=10)

# -----------------------------------------
# Panel 4: Zoomed PR Curve (Recall 0.8–1.0)
# -----------------------------------------

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(rec_cat, prec_cat, linewidth=1, color='tab:blue', label='CLS+SEQ')
ax5.plot(rec_cls, prec_cls, linewidth=1, color='tab:red', label='CLS')
ax5.plot(rec_seq, prec_seq, linewidth=1, color='tab:green', label='SEQ')
ax5.set_xlim(left=0.8, right=1.01)
ax5.set_ylim(bottom=0.4)  # let top auto-scale
ax5.set_title(label='Zoomed PR Curve (Recall 0.8–1.0)', fontsize=14)
ax5.set_xlabel(xlabel='Recall', fontsize=12)
ax5.set_ylabel(ylabel='Precision', fontsize=12)
ax5.grid(visible=True, linestyle='--', alpha=0.4)

# mark threshold on zoomed PR if mapping exists
if len(pr_thresh) > 0:
    ax5.axvline(rec_at_thr, linestyle='--', color='tab:gray', linewidth=1.5)
    ax5.plot(rec_at_thr, prec_at_thr, marker='o', markersize=6, color='tab:gray')

lines1, labels1 = ax5.get_legend_handles_labels()
ax5.legend(lines1, labels1, fontsize=11, loc='lower left')

# --------------------------------
# Panel 5: Threshold curves (full)
# --------------------------------
ax4 = fig.add_subplot(gs[2, 0])

# Accuracy (left axis)
ax4.plot(thresholds, accuracies, linewidth=2, label='Accuracy')
ax4.set_xlabel(xlabel='Decision Threshold', fontsize=12)
ax4.set_ylabel(ylabel='Accuracy', fontsize=12)
ax4.set_ylim(bottom=0, top=1)
ax4.grid(visible=True, linestyle='--', alpha=0.4)

# FP and FN counts (right axis)
ax4b = ax4.twinx()
ax4b.plot(thresholds, false_pos, linestyle='--', linewidth=2, label='False Positives', color='tab:red')
ax4b.plot(thresholds, false_neg, linestyle='--', linewidth=2, label='False Negatives', color='tab:green')
ax4b.plot(thresholds, false_neg + false_pos, linestyle='--', linewidth=2, label='False sum', color='tab:orange', visible=False)
ax4b.set_ylabel(ylabel='Counts', fontsize=12)

# vertical line at chosen threshold
ax4.axvline(THRESHOLD, linestyle='--', color='tab:gray', linewidth=1.5)

# Combined legend
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='center right')

ax4.set_title('Threshold Sweep: Accuracy, FP, FN', fontsize=16)

# -----------------------------------------
# Panel 6: Zoomed Threshold Sweep (0.3–0.8)
# -----------------------------------------
ax6 = fig.add_subplot(gs[2, 1])

mask = (thresholds >= 0.3) & (thresholds <= 0.8)
ax6.plot(thresholds[mask], accuracies[mask], linewidth=2, label='Accuracy')
ax6.set_xlabel(xlabel='Decision Threshold', fontsize=12)
ax6.set_ylabel(ylabel='Accuracy', fontsize=12)
ax6.set_ylim(bottom=0.85, top=1)
ax6.grid(visible=True, linestyle='--', alpha=0.4)

ax6b = ax6.twinx()
ax6b.plot(thresholds[mask], false_pos[mask], linestyle='--', linewidth=2, label='False Positives', color='tab:red')
ax6b.plot(thresholds[mask], false_neg[mask], linestyle='--', linewidth=2, label='False Negatives',
          color='tab:green')
ax6b.plot(thresholds[mask], false_neg[mask] + false_pos[mask], linestyle='--', linewidth=2, label='FP+FN',
          color='tab:orange', visible=True)
ax6b.set_ylabel('Counts', fontsize=12)

# Vertical line for chosen threshold if in zoom range
if 0.3 <= THRESHOLD <= 0.8:
    ax6.axvline(THRESHOLD, linestyle='--', color='tab:gray', linewidth=1.5)

# Combined legend
lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6b.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper right')

ax6.set_title(label='Zoomed Threshold Sweep (0.3–0.8)', fontsize=14)

plt.show()