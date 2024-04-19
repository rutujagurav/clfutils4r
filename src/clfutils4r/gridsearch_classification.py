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

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_objective, plot_histogram
# from sklearn.utils._testing import ignore_warnings
# from sklearn.exceptions import FitFailedWarning, ConvergenceWarning

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import collections, os, sys, random, functools, pdb, joblib, json, inspect
from pprint import pprint
from tqdm.autonotebook import tqdm


def plot_grid_search_metrics(hparam_search_results, metric="F1", show=False, save=False, save_dir=None):
    """
    Plots grid search results as a Parallel Coordinates Plot.
 
    Args:
        hparam_search_results (pandas DataFrame): This holds the results of cv_results_ from a completed run of GridSearchCV, RandomSearchCV, etc.
        metric (str): The metric to use for the colorbar in the parallel coordinates plot.
        show (bool): Whether to display the plot. Useful in notebooks.
        save (bool): Whether to save the plot as an image and html file.
        save_dir (str): If save is True, then this is the directory where the plot will be saved.
 
    Returns:
        None
    """
    
    hparams = [col for col in hparam_search_results.columns if col.startswith("param_")]
    metrics = [col for col in hparam_search_results.columns if col.startswith("mean_test_")]

    dimslist4parcoordplot = []
    for hparam_name in hparams:
        # print(hparam_name, hparam_search_results[hparam_name].dtype)
        if hparam_search_results[hparam_name].dtype in ["object", "categorical", "bool"]: 
            ## Convert None to 'None'
            hparam_search_results[hparam_name].replace({None: 'None'}, inplace=True)
            # print(hparam_search_results[hparam_name])
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
                    values=hparam_search_results[metric_name].values,
                    range=[0,1]
                )
        )

    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = hparam_search_results[f'mean_test_{metric}'],
                    colorscale = 'viridis_r',
                    showscale = True,
                    cmin=0, cmax=1
                ),
            dimensions = dimslist4parcoordplot
        )
    )
    
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fig.update_layout(width=800, height=300)
        fig.write_html(save_dir+"/parcoord_plot.html")
        fig.write_image(save_dir+"/parcoord_plot.png", scale=3)
    if show:
        fig.show()

# def to_skopt_search_space(param_grid):
#     """
#     Converts the sklearn compatible param_grid to skopt compatible search_space.

#     Args:
#         param_grid (dict): Dictionary containing the hyperparameters to search over.
    
#     Returns:
#         skopt_param_grid (dict): Dictionary containing the hyperparameters to search over in skopt format.
#     """
#     skopt_param_grid = {}
#     for key, value in param_grid.items():
#         # if the value is a list of strings, then it is a categorical parameter
#         if isinstance(value, list) and all(isinstance(i, str) for i in value):
#             skopt_param_grid[key] = Categorical(value)
#         # if the value is a list of booleans, then it is a boolean parameter
#         elif isinstance(value, list) and all(isinstance(i, bool) for i in value):
#             skopt_param_grid[key] = value
#         # if the value is a list of integers, then it is a integer parameter
#         elif isinstance(value, list) and all(isinstance(i, int) for i in value):
#             skopt_param_grid[key] = Integer(min(value), max(value))
#         # if the value is a list of floats, then it is a real parameter
#         elif isinstance(value, list) and all(isinstance(i, float) for i in value):
#             skopt_param_grid[key] = Real(min(value), max(value), prior='log-uniform')
#         else:
#             raise ValueError(f"Unknown type of parameter: {value}")
#     return skopt_param_grid
        

def classify_feats(X=None, gt_labels=[],
                    classification_algorithms=["LogisticRegression",
                                                "GaussianNB", "KNeighborsClassifier",
                                                "RandomForestClassifier", "ExtraTreesClassifier",
                                                "DecisionTreeClassifier", "ExtraTreeClassifier", 
                                            ], 
                    search_method = "RandomizedSearchCV",
                    search_space = [], # TODO: If provided, will override the classification_algorithms
                    num_runs=5, best_model_metric="F1",
                    show=False, save=False, save_dir=None
                ):
    """
    Runs a grid search for hyperparameters of various classification algorithms using specified search_method over specified search_space.

    Args:
        X (numpy array): Dataset
        gt_labels (list): Ground truth labels
        classification_algorithms (list): List of names of classification algorithms to use for grid search. If search_space is provided then this is ignored.
        search_method (str): GridSearchCV or RandomizedSearchCV
        search_space (list): List of tuples, each containing an estimator and its hyperparameter grid. If not provided then the default search space is used.
        num_runs (int): Number of cross-validation runs
        best_model_metric (str): Metric to use for selecting the best model
        show (bool): Whether to display the plot. Useful in notebooks.
        save (bool): Whether to save the outputs.
        save_dir (str): If save is True, then this is the directory where the outputs will be saved.
    
    Returns:
        best_estimator_overall (sklearn estimator): The best model found during the grid search
        grid_search_results (dict): A dictionary containing the results of the grid search for each estimator
    
    """
    num_samples = X.shape[0]
    num_classes = len(np.unique(gt_labels))
    print(f"No. of classes: {num_classes}")
    print(f"Class counts: {collections.Counter(gt_labels)}")

    def scorer(estimator, X, y_true):
        estimator.fit(X, y_true)
        y_pred = estimator.predict(X)
        # y_pred_proba = estimator.predict_proba(X)
        if num_classes == 2:
            return {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, average='binary'),
                    "Recall": recall_score(y_true, y_pred, average='binary'),
                    "F1": f1_score(y_true, y_pred, average='binary'),
                    "AUROC": roc_auc_score(y_true, y_pred), #"AUROC": roc_auc_score(y_true, y_pred_proba[:,1]),
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
                    "AUROC": roc_auc_score(y_true, y_pred, multi_class='ovr'), #"AUROC": roc_auc_score(y_true, y_pred_proba, multi_class='ovr'),
                    # "Discounted_Cumulative_Gain": dcg_score(y_true, y_pred),
                    # "Normalized_Discounted_Cumulative_Gain": ndcg_score(y_true, y_pred),
                    # "Cohen_Kappa": cohen_kappa_score(y_true, y_pred)
                }
    
    if len(search_space) == 0:
        search_space = []
        if "LogisticRegression" in classification_algorithms:
            search_space.append([LogisticRegression(), {
                                                            "penalty": ['l1', 'l2', 'elasticnet'], 
                                                            "fit_intercept": [True, False],
                                                            "C": [1,2,4]
                                                    }])
        if "RidgeClassifier" in classification_algorithms:
            search_space.append([RidgeClassifier(), {
                                                        "alpha": [1,2,4],
                                                        "fit_intercept": [True, False],
                                                    }])
        if "DecisionTreeClassifier" in classification_algorithms:
            min_samples_split = np.round(np.linspace(2/num_samples, 0.01, 10), 5)
            # print(f"min_samples_split: {min_samples_split}")
            search_space.append([DecisionTreeClassifier(), {
                                                                "criterion": ['gini', 'entropy', 'log_loss'],
                                                                "splitter": ['best', 'random'],
                                                                "min_samples_split": min_samples_split, #[0.01, 0.05, 0.1],
                                                                "class_weight": ['balanced', None],
                                                                # "monotonic_cst": [None, [1,0], [0,1], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]]
                                                            }])
        if "ExtraTreeClassifier" in classification_algorithms:
            min_samples_split = np.round(np.linspace(2/num_samples, 0.01, 10), 5)
            # print(f"min_samples_split: {min_samples_split}")
            search_space.append([ExtraTreeClassifier(), {
                                                            "criterion": ['gini', 'entropy', 'log_loss'],
                                                            "splitter": ['best', 'random'],
                                                            "min_samples_split": min_samples_split, #[0.01, 0.05, 0.1],
                                                            "class_weight": ['balanced', None],
                                                        }])
        if "GaussianNB" in classification_algorithms:
            search_space.append([GaussianNB(), {"var_smoothing": [1e-9, 1e-5, 1e-3]}])

        if "KNeighborsClassifier" in classification_algorithms:
            search_space.append([KNeighborsClassifier(), {
                                                            "n_neighbors": [1, 5, 10],
                                                            "weights": ['uniform', 'distance'],
                                                            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                                            "p": [1,2]
                                                        }])
        if "RandomForestClassifier" in classification_algorithms:
            min_samples_split = np.round(np.linspace(2/num_samples, 0.01, 10), 5)
            # print(f"min_samples_split: {min_samples_split}")
            search_space.append([RandomForestClassifier(), {
                                                            "n_estimators": [10, 50, 100, 200, 500],
                                                            "criterion": ['gini', 'entropy', 'log_loss'],
                                                            "bootstrap": [True, False],
                                                            "min_samples_split": min_samples_split, #[0.01, 0.05, 0.1],
                                                            "max_features": ['sqrt', 'log2', None],
                                                        }])
        if "ExtraTreesClassifier" in classification_algorithms:
            min_samples_split = np.round(np.linspace(2/num_samples, 0.01, 10), 5)            
            # print(f"min_samples_split: {min_samples_split}")
            search_space.append([ExtraTreesClassifier(), {
                                                            "n_estimators": [10, 50, 100, 200, 500],
                                                            "criterion": ['gini', 'entropy', 'log_loss'],
                                                            "bootstrap": [True, False],
                                                            "min_samples_split": min_samples_split, #[0.01, 0.05, 0.1],
                                                        }])
        if "GaussianProcessClassifier" in classification_algorithms:
            search_space.append([GaussianProcessClassifier(), {}])
        if "LinearSVC" in classification_algorithms:
            search_space.append([LinearSVC(), {
                                                "penalty": ['l1', 'l2'],
                                                "loss": ['hinge', 'squared_hinge'],
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

        
        if search_method == "GridSearchCV":
            print(f"Using GridSearchCV for {estimator_name}...")
            classifier_hpopt = GridSearchCV(estimator=estimator,
                                            param_grid=param_grid,
                                            scoring=scorer,
                                            refit=best_model_metric,
                                            cv=num_runs, n_jobs=-1,
                                            error_score=np.nan, ## Skip forbidden parameter settings
                                            verbose=10,
                                        )
        elif search_method == "RandomizedSearchCV":
            print(f"Using RandomizedSearchCV for {estimator_name}...")
            classifier_hpopt = RandomizedSearchCV(estimator=estimator,
                                                param_distributions=param_grid,
                                                scoring=scorer,
                                                n_iter=50,
                                                refit=best_model_metric,
                                                cv=num_runs, n_jobs=-1,
                                                error_score=np.nan, ## Skip forbidden parameter settings
                                                verbose=0,
                                            )
        # elif search_method == "BayesSearchCV":
        #     print(f"Using BayesSearchCV for {estimator_name}...")
        #     param_grid = to_skopt_search_space(param_grid)
        #     classifier_hpopt = BayesSearchCV(estimator=estimator,
        #                                     search_spaces=param_grid,
        #                                     scoring=scorer,
        #                                     n_iter=10,
        #                                     refit=best_model_metric,
        #                                     cv=num_runs, n_jobs=-1,
        #                                     error_score=np.nan, ## Skip forbidden parameter settings: Not working for BayesSearchCV
        #                                     verbose=1,
        #                                 )
        else:
            raise ValueError(f"Unknown search method: {search_method}")
            
        classifier_hpopt.fit(X, gt_labels)

        best_model = classifier_hpopt.best_estimator_

        # Check if current score is better than best score and update if true
        if classifier_hpopt.best_score_ > best_score_overall:
            best_score_overall = classifier_hpopt.best_score_
            best_estimator_overall = best_model
        
        grid_search_results_df = pd.DataFrame(classifier_hpopt.cv_results_) #.fillna(0)
        grid_search_results[estimator_name] = grid_search_results_df
        # print(grid_search_results_df.columns)
        
        ## Plot grid search results
        plot_grid_search_metrics(grid_search_results_df, 
                                 metric=best_model_metric,
                                 show=show, save=save, save_dir=save_dir_)
        # if search_method == "BayesSearchCV":
        #     # print(list(param_grid.keys()), classifier_hpopt.optimizer_results_[0])
        #     print(classifier_hpopt.optimizer_results_)
        #     fig, ax = plt.subplots(figsize=(10,10))
        #     plot = plot_objective(classifier_hpopt.optimizer_results_[0],
        #                             # dimensions=list(param_grid.keys()),
        #                             ax=ax
        #                         )
        #     plt.tight_layout()
        #     if save: plt.savefig(os.path.join(save_dir_, 'bayes_search_plot.png'))
        #     if show: plt.show()
        #     plt.close()
        
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

    return best_estimator_overall, best_score_overall, grid_search_results

gridsearch_classification = classify_feats

if __name__ == "__main__":
    '''Test the classify_feats function'''

    rng = np.random.RandomState(0)
    n_samples=1000

    ### Synthetic data
    X, y = make_classification(n_samples=n_samples, n_features=20, n_redundant=3, n_repeated=1, n_informative=10, n_classes=2, random_state=42)

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