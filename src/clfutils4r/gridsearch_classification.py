import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import sklearn
print(f"sklearn version: {sklearn.__version__}")
# from sklearn.externals.joblib import parallel_backend

from sklearn.manifold import Isomap, SpectralEmbedding
from sklearn.manifold import TSNE
# from openTSNE import TSNE
from sklearn.decomposition import PCA

from sklearn.datasets import make_classification, load_iris, load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

## Classification models
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, dcg_score, ndcg_score, cohen_kappa_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import collections, os, sys, random, functools, pdb, joblib, json, inspect
from pprint import pprint
from tqdm.autonotebook import tqdm


def plot_grid_search_metrics(hparam_search_results, metric="F1", show=False, save=False, save_dir=None):
    hparams = [col for col in hparam_search_results.columns if col.startswith("param_")]
    metrics = [col for col in hparam_search_results.columns if col.startswith("mean_test_")]

    dimslist4parcoordplot = []
    for hparam_name in hparams:
        if hparam_search_results[hparam_name].dtype in ["object", "categorical"]: 
            le = LabelEncoder()
            encoded_hparam = le.fit_transform(hparam_search_results[hparam_name])
            dimslist4parcoordplot.append(
                dict(   
                        ticktext=le.classes_, tickvals=le.transform(le.classes_),
                        label=hparam_name.split("param_")[-1], 
                        values=encoded_hparam
                    )
            )
        else:
            dimslist4parcoordplot.append(
                dict(
                        label=hparam_name.split("param_")[-1], 
                        values=hparam_search_results[hparam_name].values
                    )
            )
    for metric_name in metrics:
        dimslist4parcoordplot.append(
            dict(
                    label=metric_name.split("mean_test_")[-1], 
                    values=hparam_search_results[metric_name].values
                )
        )

    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = hparam_search_results[f'mean_test_{metric}'],
                    colorscale = 'Viridis',
                    showscale = True,
                ),
            dimensions = dimslist4parcoordplot
        )
    )
    # fig.update_layout(width=1500, height=500)
    if save:
        fig.write_html(save_dir+"/parcoord_plot.html")
        fig.write_image(save_dir+"/parcoord_plot.png")
    if show:
        fig.show()

def classify_feats(X=None, gt_labels=[],
                    classification_algorithms=["LogisticRegression",
                                                "GaussianNB", "KNeighborsClassifier",
                                                "DecisionTreeClassifier", "ExtraTreeClassifier", 
                                                "RandomForestClassifier", "ExtraTreesClassifier",
                                            ], 
                    num_runs=5, best_model_metric="F1",
                    show=False, save=False, save_dir=None
                ):
    num_classes = len(np.unique(gt_labels))
    print(f"No. of classes: {num_classes}")
    print(f"Class counts: {collections.Counter(gt_labels)}")

    def scorer(estimator, X, y_true):
        estimator.fit(X, y_true)
        y_pred = estimator.predict(X)
        y_pred_proba = estimator.predict_proba(X)
        if num_classes == 2:
            return {
                    "Accuracy": accuracy_score(y_true, y_pred > 0.5),
                    "Precision": precision_score(y_true, y_pred > 0.5),
                    "Recall": recall_score(y_true, y_pred > 0.5),
                    "F1": f1_score(y_true, y_pred > 0.5),
                    "AUROC": roc_auc_score(y_true, y_pred),
                    # "Discounted_Cumulative_Gain": dcg_score(y_true, y_pred),
                    # "Normalized_Discounted_Cumulative_Gain": ndcg_score(y_true, y_pred),
                    # "Cohen_Kappa": cohen_kappa_score(y_true, y_pred)
                }
        elif num_classes > 2:
            return {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, average='weighted'),
                    "Recall": recall_score(y_true, y_pred, average='weighted'),
                    "F1": f1_score(y_true, y_pred, average='weighted'),
                    "AUROC": roc_auc_score(y_true, y_pred_proba, multi_class='ovr'),
                    # "Discounted_Cumulative_Gain": dcg_score(y_true, y_pred),
                    # "Normalized_Discounted_Cumulative_Gain": ndcg_score(y_true, y_pred),
                    # "Cohen_Kappa": cohen_kappa_score(y_true, y_pred)
                }
    
    search_space = []
    if "LogisticRegression" in classification_algorithms:
        search_space.append([LogisticRegression(), {
                                                        "penalty": ['l2', 'elasticnet'],
                                                        "fit_intercept": [True, False],
                                                        "C": [1,2,4]
                                                }])
    if "RidgeClassifier" in classification_algorithms:
        search_space.append([RidgeClassifier(), {
                                                    "alpha": [1,2,4],
                                                    "fit_intercept": [True, False],
                                                }])
    if "DecisionTreeClassifier" in classification_algorithms:
        search_space.append([DecisionTreeClassifier(), {
                                                            "criterion": ['gini', 'entropy', 'log_loss'],
                                                            "splitter": ['best', 'random']
                                                        }])
    if "ExtraTreeClassifier" in classification_algorithms:
        search_space.append([ExtraTreeClassifier(), {
                                                        "criterion": ['gini', 'entropy', 'log_loss'],
                                                        "splitter": ['best', 'random']

                                                    }])
    if "GaussianNB" in classification_algorithms:
        search_space.append([GaussianNB(), {}])

    if "KNeighborsClassifier" in classification_algorithms:
        search_space.append([KNeighborsClassifier(), {
                                                        "n_neighbors": [1, 5, 10],
                                                        "weights": ['uniform', 'distance'],
                                                        "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                                        "p": [1,2]
                                                    }])
    if "RandomForestClassifier" in classification_algorithms:
        search_space.append([RandomForestClassifier(), {
                                                        "n_estimators": [10, 50, 100, 200],
                                                        "criterion": ['gini', 'entropy', 'log_loss'],
                                                        "bootstrap": [True, False]
                                                    }])
    if "ExtraTreesClassifier" in classification_algorithms:
        search_space.append([ExtraTreesClassifier(), {
                                                        "n_estimators": [10, 50, 100, 200],
                                                        "criterion": ['gini', 'entropy', 'log_loss'],
                                                        "bootstrap": [True, False]
                                                    }])
    if "GaussianProcessClassifier" in classification_algorithms:
        search_space.append([GaussianProcessClassifier(), {}])
    if "LinearSVC" in classification_algorithms:
        search_space.append([LinearSVC(), {
                                            "penalty": ['l1', 'l2', 'elasticnet'],
                                            "C": [1,2,4],
                                            "fit_intercept": [True, False]
                                        }])    
    
    print("Search space:")
    pprint(search_space)
    print()

    grid_search_results={}
    best_score_overall = float('-inf') # Initialize with the worst possible score
    best_estimator_overall = None
    for estimator, param_grid in tqdm(search_space):
        estimator_name = estimator.__class__.__name__

        print(f"Searching for best hyperparameters for {estimator_name}...")
        print(f"Available parameters: {list(estimator.get_params().keys())}")
        print(f"But only searching for parameters: {list(param_grid.keys())}")

        if save:
            save_dir_ = os.path.join(save_dir, "models", estimator_name)
            if not os.path.exists(save_dir_): os.makedirs(save_dir_)
        else:
            save_dir_ = None

        classifier_hpopt = GridSearchCV(estimator=estimator,
                                        param_grid=param_grid,
                                        scoring=scorer,
                                        refit=best_model_metric,
                                        cv=num_runs, n_jobs=-1,
                                        verbose=0,
                                    )
        classifier_hpopt.fit(X, gt_labels)

        best_model = classifier_hpopt.best_estimator_

        # Check if current score is better than best score and update if true
        if classifier_hpopt.best_score_ > best_score_overall:
            best_score_overall = classifier_hpopt.best_score_
            best_estimator_overall = best_model
        
        grid_search_results_df = pd.DataFrame(classifier_hpopt.cv_results_).fillna(0)
        grid_search_results[estimator_name] = grid_search_results_df
        # print(grid_search_results_df.columns)
        
        ## Plot grid search results
        plot_grid_search_metrics(grid_search_results_df, 
                                 metric=best_model_metric,
                                 show=show, save=save, save_dir=save_dir_)
        
        if save:
            ## Save cv_results_ to a json file
            grid_search_results_df.to_json(os.path.join(save_dir_, 'grid_search_results.json'))
            grid_search_results_df.to_csv(os.path.join(save_dir_, 'grid_search_results.csv'), index=False)

            ## Save best model params
            best_model_info = {"model_name":estimator_name, "model_params": best_model.get_params()}
            print()
            with open(os.path.join(save_dir_, 'best_model_info.json'), 'w') as f:
                json.dump(best_model_info, f, default=str)

            ## Save the best models using joblib
            joblib.dump(best_model, os.path.join(save_dir_, f'best_model__{estimator_name}.joblib'))
    
    if save:
        ## Save best overall model params
        best_model_info = {"model_name":best_estimator_overall.__class__.__name__, "model_params": best_estimator_overall.get_params()}
        with open(os.path.join(save_dir, 'best_model_info.json'), 'w') as f:
            json.dump(best_model_info, f, default=str)
        ## Save the best overall model using joblib
        joblib.dump(best_estimator_overall, os.path.join(save_dir, f'best_model.joblib'))

    return best_estimator_overall, grid_search_results

gridsearch_classification = classify_feats

if __name__ == "__main__":
    ## For testing purposes

    rng = np.random.RandomState(0)
    n_samples=1000

    ### Synthetic data
    X, y = make_classification(n_samples=n_samples, n_classes=5)

    ### Real benchmark
    # data = load_iris()
    # X, y = data.data, data.target

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    best_model = classify_feats(
                                X=X, gt_labels=y,
                                num_runs=10, best_model_metric="F1",
                                show=False, save=True, save_dir=save_dir,
                            )
    print("Best model: ", best_model.__class__.__name__)
    print("Best model params: ", best_model.get_params())