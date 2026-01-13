import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import probplot
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression, RidgeCV, Ridge, ElasticNetCV
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score
from textwrap import fill
import lightgbm as lgb

CV_NUM = 5 # Number of cross-validation splits to perform
RANDOM_STATE = 42
EMBEDDINGS_FILE_PATH = 'V2_2_5B_multi_PF_geno_pheno_reconstructed.h5'
THRESHOLD = 0.5
LAYERS_TO_ANALYZE = range(11, 12)  # list here as tuple or range. Add a trailing comma if only one item
# For all layers, set to range(int(len(layer_keys)/2)) (if you saved every layer from the model)

def make_weights(y, threshold=2.5, high_weight=3.0):
    """
    Emphasize extreme log2 fold changes.
    """
    w = np.ones_like(y, dtype=float)
    w[np.abs(y) >= threshold] = high_weight
    return w

class NoOpTransformer(TransformerMixin):
    """
    Defines a sci-kit learn transformer that does nothing.
    This allows code that uses a transformer (e.g., Yeo-Johnson) or not without
    having to modify the code base other than specifying which transformer to use.
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def inverse_transform(self, X):
        return X

# ------------------------------------
# Open HDF5 file and determine datasets
# ------------------------------------

with (h5py.File(EMBEDDINGS_FILE_PATH, mode='r') as ifh):
    print(f'Datasets present in the h5 file {EMBEDDINGS_FILE_PATH}:')
    print(fill(', '.join(ifh.keys()), width=120))
    layer_keys = [k for k in ifh.keys() if k not in ('Labels', 'Log2fold')]

    if 'Log2fold' in ifh.keys():
        all_fold_change = np.array(ifh['Log2fold'])
        print(f'Length of fold change array: {len(all_fold_change)}')

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

        print(f'\nSplitting and training {len(X_cat)} embeddings from layer {layer_index}:')
        X = X_seq
        print(f'Shape of X is {X.shape}')
        y = all_fold_change

        # # ----------------------
        # # Ridge regression
        # # ----------------------
        #
        # model = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('ridge', RidgeCV(
        #         alphas=np.logspace(start=-0.5, stop=4, num=50),
        #         cv=CV_NUM
        #     ))
        # ])
        #
        # model.fit(X_train, y_train)  # Fit
        #
        # y_pred = model.predict(X_test)  # Predict
        #
        # # Evaluation
        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # r2 = r2_score(y_test, y_pred)
        #
        # print('Ridge Regression:')
        # best_alpha = model.named_steps['ridge'].alpha_
        # print("Best alpha:", best_alpha)
        # print("RMSE:", rmse)
        # print("R^2:", r2)
        #
        # # Retrain with the best alpha
        # final_model = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('ridge', Ridge(alpha=best_alpha))
        # ])
        #
        # final_model.fit(X_train, y_train)
        # y_pred = final_model.predict(X_test)  # Careful this is redefined by LightGBM

        # ----------------------
        # Elastic Net regression
        # ----------------------
        # model = Pipeline([
        #     ('scaler', StandardScaler()),  # scale features
        #     ('elasticnet', ElasticNetCV(
        #         l1_ratio=[0.01, 0.05, 0.1, 0.5, 0.9],  # L1 vs L2 balance
        #         alphas=np.logspace(start=-0.75, stop=4, num=50),  # candidate alpha values
        #         cv=CV_NUM,  # 5-fold cross-validation
        #         max_iter=25000,
        #         random_state=RANDOM_STATE
        #     ))
        # ])
        #
        # model.fit(X_train, y_train)  # Fit model
        # y_pred = model.predict(X_test) # Predict
        #
        # # Metrics
        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # r2 = r2_score(y_test, y_pred)
        # print('Elastic Net Regression:')
        # print("Best alpha:", model.named_steps['elasticnet'].alpha_)
        # print("Best l1_ratio:", model.named_steps['elasticnet'].l1_ratio_)
        # print("RMSE:", rmse)
        # print("R^2:", r2)

        # --------------
        # LightGBM model
        # --------------
        params = {
            "objective": "huber", # options huber, regression, regression_l1
            "alpha": 0.25,
            "metric": "rmse", # 'mae' is probably more consistent with huber as the objective than rmse
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 47,
            "max_depth": -1,  # let num_leaves control complexity
            "min_data_in_leaf": 10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": RANDOM_STATE
        }
        # LightGBM CVs
        rmse_list = []
        mae_list = []
        best_iters = []

        y_orig = y.copy()

        kf = KFold(n_splits=CV_NUM, shuffle=True, random_state=RANDOM_STATE)

        for train_idx, test_idx in kf.split(X):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y_orig[train_idx]
            y_test = y_orig[test_idx]

            # Fit transformer on training target only
            pt = NoOpTransformer()
            # Optionally, pt=PowerTransformer(method='yeo-johnson')
            y_train_trans = pt.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_trans = pt.transform(y_test.reshape(-1, 1)).flatten()

            train_weights = make_weights(y_train, threshold=5000, high_weight=5000)

            lgb_train = lgb.Dataset(X_train, y_train_trans, weight=train_weights)
            lgb_valid = lgb.Dataset(X_test, y_test_trans, reference=lgb_train)

            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=[lgb_valid],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=False)
                ],
            )

            y_pred_trans = model.predict(X_test, num_iteration=model.best_iteration)

            # Inverse transform to original space
            y_pred = pt.inverse_transform(y_pred_trans.reshape(-1, 1)).flatten()
            y_test_orig = y_test  # already in the original log2 fold-change space

            rmse_list.append(np.sqrt(mean_squared_error(y_test_orig, y_pred)))
            mae_list.append(np.mean(np.abs(y_test_orig - y_pred)))
            best_iters.append(model.best_iteration)

        rmse_mean = np.mean(rmse_list)
        rmse_std = np.std(rmse_list)
        mae_mean = np.mean(mae_list)
        mae_std = np.std(mae_list)

        print(f'{CV_NUM}-fold CV RMSE: {rmse_mean:.3f} ± {rmse_std:.3f}')
        print(f'{CV_NUM}-fold CV MAE: {mae_mean:.3f} ± {mae_std:.3f}')

        # # -------------------------
        # # Retrain on full dataset
        # # -------------------------
        # # Uncomment this block to
        # print('Retraining on full dataset:')
        #
        # X_full = X
        # y_full = y_orig
        #
        # # Transform the full target if needed
        # y_full_trans = pt.fit_transform(y_full.reshape(-1, 1)).flatten()
        #
        # train_weights_full = make_weights(y_full, threshold=5000, high_weight=5000)
        # lgb_full = lgb.Dataset(X_full, y_full_trans, weight=train_weights_full)
        #
        # final_model = lgb.train(
        #     params,
        #     lgb_full,
        #     num_boost_round= np.mean(best_iters).astype(int),
        #     valid_sets=[lgb_full],  # no separate validation
        #     callbacks=[lgb.log_evaluation(period=False)]
        # )
        #
        # # Predictions on the full dataset
        # y_pred_full_trans = final_model.predict(X_full, num_iteration=final_model.best_iteration)
        # y_pred = pt.inverse_transform(y_pred_full_trans.reshape(-1, 1)).flatten()
        # y_test = y_full.copy()  # original targets for plotting
        #
        # # model = lgb.train(
        # #     params,
        # #     lgb_train_data,
        # #     num_boost_round=2000,
        # #     valid_sets=[lgb_test_data],
        # #     callbacks=[
        # #         lgb.early_stopping(stopping_rounds=100),
        # #         lgb.log_evaluation(period=0)
        # #     ]
        # # )
        # #
        # # # Predictions
        # # y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        # #
        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # r2 = r2_score(y_test, y_pred)
        #
        #
        # print(f"RMSE: {rmse:.3f}")
        # print(f"R^2: {r2:.3f}")
        # print(f"Best iteration: {model.best_iteration}")

        # -----------------------------
        # Create a figure with 4 panels
        # -----------------------------

        residuals = y_test - y_pred  # Compute residuals
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()  # Flatten 2x2 array to 1D for easy indexing
        # Panel 1 Predicted vs. Observed
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        axes[0].text(0.05, 0.95, f'RMSE={rmse:.2f}\nR²={r2:.2f}', transform=axes[0].transAxes,
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        sns.scatterplot(x=y_test, y=y_pred, ax=axes[0])
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect prediction')
        axes[0].set_xlabel('Observed Log2 Fold Change')
        axes[0].set_ylabel('Predicted Log2 Fold Change')
        axes[0].set_title('Predicted vs Observed')
        axes[0].legend()

        # Panel 2 Residuals histogram
        sns.histplot(residuals, kde=True, bins=30, ax=axes[1])
        axes[1].set_xlabel('Residuals (Observed - Predicted)')
        axes[1].set_title('Residuals Distribution')

        # Panel 3 Residuals vs. Predicted
        sns.scatterplot(x=y_pred, y=residuals, ax=axes[2])
        axes[2].axhline(0, color='r', linestyle='--')
        axes[2].set_xlabel('Predicted Log2 Fold Change')
        axes[2].set_ylabel('Residuals')
        axes[2].set_title('Residuals vs Predicted')

        probplot(residuals, dist="norm", plot=axes[3])
        axes[3].set_title('Q-Q Plot of Residuals')

        # Let's take a look!
        plt.tight_layout()
        plt.show()
