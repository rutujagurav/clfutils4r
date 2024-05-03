import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.facecolor'] = plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.labelsize'] = \
    plt.rcParams['axes.titlesize'] = \
    plt.rcParams['xtick.labelsize'] = \
    plt.rcParams['ytick.labelsize'] = \
    'xx-large'

# from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
# sns.set()
import pandas as pd
import numpy as np
import os, textwrap
from deprecation import deprecated

import sklearn
print(f"sklearn version: {sklearn.__version__}")
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, auc, classification_report, roc_curve, precision_recall_curve
from sklearn.manifold import TSNE

import scikitplot
print(f"scikitplot version: {scikitplot.__version__}")
from . import pretty_cm

import shap

dpi=300

@deprecated
def print_metrics(y_test=None, y_pred=None, threshold = 0.5):
    print("Precision", precision_score(y_test, y_pred > threshold))
    print("Recall",recall_score(y_test, y_pred > threshold))
    print("F1",f1_score(y_test, y_pred > threshold))
    print("AUC",roc_auc_score(y_test, y_pred)) 

@deprecated
def plot_PR_curve(y_test=None, y_pred=None, RESULTS_DIR='', show=False, save=False, dpi=300):
    y_score = y_pred
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_pred)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_pred >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set(ylim=(0, 1))

    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax1.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    plt.title('Precision Recall Curve')
    
    if(save):
        plt.savefig(RESULTS_DIR+'/precision_recall_curve.png', bbox_inches='tight',dpi=dpi)
    if(show):
        plt.show()
    plt.close()

@deprecated
def plot_ROC_curve(y_test=None, y_pred=None, RESULTS_DIR='', show=False, save=False, dpi=300):
    fpr,tpr,thresholds = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    plt.clf()
    plt.title("ROC AUC")
    plt.plot(fpr,tpr,'b',label='AUC = %0.2f' %roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    if(save):
        plt.savefig(RESULTS_DIR+'/roc_curve.png', bbox_inches='tight',dpi=dpi)
    if(show):
        plt.show()
    plt.close()

def plot_confusion_matrix(y_test=None, y_pred=None, labels=[], threshold = 0.5, RESULTS_DIR='', titlestr='', pretty=True, show=False, save=False, dpi=300):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       y_test: ground-truth labels
       y_pred: predicted labels
       labels: class_names since y_test, y_pred are encoded.
    """
    num_classes = len(set(y_test))
    if num_classes <= 3: figsize=(5,5)
    else: figsize=(num_classes, num_classes)

    if not pretty:
        ## Create confusion matrix using scikit-plot
        fig, ax = plt.subplots()
        ax = scikitplot.metrics.plot_confusion_matrix(y_test, y_pred, 
                                                    normalize=True, 
                                                    text_fontsize='x-large', 
                                                    title_fontsize='xx-large',
                                                    ax=ax)
        if(len(labels) != 0):
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(labels, rotation=0, ha="center", rotation_mode="anchor")
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(labels, rotation=90, ha="center", rotation_mode="anchor")
    else:
        cm = confusion_matrix(y_test, y_pred)
        pretty_cm.plot_from_confusion_matrix(cm, columns=labels, 
                                                fz='x-large', 
                                                figsize=figsize
                                            )
    # ## Create confusion matrix using sklearn
    # cm = confusion_matrix(y_test, y_pred)
    # # Normalize the confusion matrix.
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    # # print(cm.shape)

    # vals = [["{}\n(True Positives)".format(str(cm[0,0])), "{}\n(False Positives)".format(str(cm[0,1]))],
    #         ["{}\n(False Negatives)".format(str(cm[1,0])), "{}\n(True Negatives)".format(str(cm[0,1]))]
    #     ]
    # print(vals)
    
    # fig, ax = plt.subplots()
    # s = sns.heatmap(cm, vmin=0, vmax=1, center=0.5, 
    #                 xticklabels = labels, yticklabels = labels,
    #                 cmap=plt.cm.Blues, cbar=False, 
    #                 ax=ax, square=True, 
    #                 annot=True) # , annot_kws={"fontsize":10}
    # s.set_ylabel('True Label')
    # s.set_xlabel('Predicted Label')

    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion matrix")
    # plt.colorbar()
    # tick_marks = np.arange(len(class_names))+0.5
    # ax.set_xticks(tick_marks, class_names, rotation=90)
    # ax.set_yticks(tick_marks, class_names)
    ### Use white text if squares are dark; otherwise black.
    # threshold = cm.max() / 2.0
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     color = "white" if cm[i, j] > threshold else "black"
    #     ax.set_text(j, i, cm[i, j], horizontalalignment="center", color=color)
    # ax.set_ylabel('True Label', fontsize=15)
    # ax.set_xlabel('Predicted Label', fontsize=15)
        
    # plt.title(titlestr+"Confusion Matrix")


    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR+'/confusion_matrix.png', bbox_inches='tight',dpi=dpi)
    if show:
        plt.show()
    plt.close()
        
def plot_ks_stat(y_test=None, y_pred=None, titlestr='', classes_titlestr='', RESULTS_DIR='', show=False, save=False, dpi=300):
    if len(classes_titlestr) > 50: classes_titlestr = '\n'.join(textwrap.wrap(classes_titlestr, 50))

    fig, ax = plt.subplots(figsize=(10,5))
    ax = scikitplot.metrics.plot_ks_statistic(y_test, y_pred, 
                                            text_fontsize='xx-large', 
                                            title_fontsize='xx-large',
                                            ax=ax)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_facecolor('white')
    for l in ax.lines:
        l.set_lw(2)
    plt.title(titlestr+"KS Statistic Plot")
    plt.grid()
    legend = plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', title=classes_titlestr, fontsize='large')
    legend.get_title().set_fontsize('large')
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR+'/ks_stat.png', bbox_inches='tight',dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_lift_curve(y_test=None, y_pred=None, titlestr='', classes_titlestr='', RESULTS_DIR='', show=False, save=False, dpi=300):
    if len(classes_titlestr) > 50: classes_titlestr = '\n'.join(textwrap.wrap(classes_titlestr, 50))

    fig, ax = plt.subplots(figsize=(10,5))
    ax = scikitplot.metrics.plot_lift_curve(y_test, y_pred, 
                                            text_fontsize='xx-large', 
                                            title_fontsize='xx-large',
                                            ax=ax)
    ax.set_facecolor('white')
    for l in ax.lines:
        l.set_lw(2)
    plt.title(titlestr+"Lift Curve", 
            #   fontsize=axes_label_fontsize
            )
    # plt.grid()
    legend = plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', title=classes_titlestr, fontsize='large')
    legend.get_title().set_fontsize('large')
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR+'/lift_curve.png', bbox_inches='tight',dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_cumul_gain(y_test=None, y_pred=None, titlestr='', classes_titlestr='', RESULTS_DIR='', show=False, save=False, dpi=300):
    if len(classes_titlestr) > 50: classes_titlestr = '\n'.join(textwrap.wrap(classes_titlestr, 50))

    fig, ax = plt.subplots(figsize=(10,5))
    ax = scikitplot.metrics.plot_cumulative_gain(y_test, y_pred, 
                                                text_fontsize='xx-large', 
                                                title_fontsize='xx-large',
                                                ax=ax)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_facecolor('white')
    for l in ax.lines:
        l.set_lw(2)
    plt.title(titlestr+"Cumulative Gains Curve")
    # plt.grid()
    legend = plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', title=classes_titlestr, fontsize='large')
    legend.get_title().set_fontsize('large')
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR+'/cumul_gain.png', bbox_inches='tight',dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_classwise_pr_curve(y_test=None, y_pred=None, titlestr='', classes_titlestr='', RESULTS_DIR='', show=False, save=False, dpi=300):
    if len(classes_titlestr) > 50: classes_titlestr = '\n'.join(textwrap.wrap(classes_titlestr, 50))
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax = scikitplot.metrics.plot_precision_recall(y_test, y_pred, 
                                                # text_fontsize='xx-large', 
                                                title_fontsize='xx-large',
                                                ax=ax)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_facecolor('white')
    for l in ax.lines:
        l.set_lw(2)
    ax.set_xticklabels(ax.get_xticks().round(1), fontsize='x-large')
    ax.set_yticklabels(ax.get_yticks().round(1), fontsize='x-large')
    ax.set_xlabel(ax.get_xlabel(), fontsize='xx-large')
    ax.set_ylabel(ax.get_ylabel(), fontsize='xx-large')
    plt.title(titlestr+"Precision-Recall Curve")
    plt.grid()
    legend = plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', title=classes_titlestr, fontsize='large')
    legend.get_title().set_fontsize('large')
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR+'/classwise_pr_curve.png', bbox_inches='tight',dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_classwise_roc_curve(y_test=None, y_pred=None, titlestr='', classes_titlestr='', RESULTS_DIR='', show=False, save=False, dpi=300):
    if len(classes_titlestr) > 50: classes_titlestr = '\n'.join(textwrap.wrap(classes_titlestr, 50))

    fig, ax = plt.subplots(figsize=(10,5))
    ax = scikitplot.metrics.plot_roc(y_test, y_pred, 
                                    # text_fontsize='xx-large', 
                                    title_fontsize='xx-large',
                                    ax=ax)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_facecolor('white')
    ax.set_xticklabels(ax.get_xticks().round(1), fontsize='x-large')
    ax.set_yticklabels(ax.get_yticks().round(1), fontsize='x-large')
    ax.set_xlabel(ax.get_xlabel(), fontsize='xx-large')
    ax.set_ylabel(ax.get_ylabel(), fontsize='xx-large')
    for l in ax.lines:
        l.set_lw(2)
    plt.title(titlestr+"ROC Curves")
    plt.grid()
    legend = plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', title=classes_titlestr, fontsize='large')
    legend.get_title().set_fontsize('large')
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR+'/classwise_roc_curve.png', bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    plt.close()


def generate_classification_report(y_test=None, y_pred=None, labels=[], RESULTS_DIR='', show=False, save=False):
    if(len(labels) !=0):
        report = classification_report(y_test, 
                                        y_pred, 
                                        target_names=[cl+'(class '+str(i)+')' for i, cl in enumerate(labels)], 
                                        output_dict=True)
    else:
        report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    if save:
        report_df.to_csv(RESULTS_DIR+'/classification_report.csv')
    if show:
        print(report_df)

def plot_cv_roc_curve(classifier=None, cv=None, n_splits=10, X=None, y=None, RESULTS_DIR='', titlestr='', show=False, save=False, dpi=300):
    """
    Draw a Cross Validated ROC Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Features Pandas DataFrame
        y: Target Pandas Series
    Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(n_splits=n_splits)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(titlestr+'Cross-Validated ROC Curve')
    plt.legend(bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    plt.grid()
    
    if(save):
        plt.savefig(RESULTS_DIR+'/crossvalidation_roc_curve.png', bbox_inches='tight',dpi=dpi)
    if(show):
        plt.show()
    plt.close()

def plot_cv_pr_curve(classifier=None, cv=None, n_splits=10, X=None, y=None, RESULTS_DIR='', titlestr='', show=False, save=False, dpi=300):
    """
    Draw a Cross Validated PR Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series

    Largely taken from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    """
    y_real = []
    y_proba = []
    cv = StratifiedKFold(n_splits=n_splits)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        precision, recall, _ = precision_recall_curve(y.iloc[test], probas_[:, 1])

        # Plotting each individual PR Curve
        plt.plot(recall, precision, lw=1, alpha=0.3,
                 label='PR fold %d (AP = %0.2f)' % (i, average_precision_score(y.iloc[test], probas_[:, 1])))

        y_real.append(y.iloc[test])
        y_proba.append(probas_[:, 1])

        i += 1

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    plt.plot(recall, precision, color='b',
             label=r'Average Precision (AP = %0.2f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(titlestr+'Cross-Validated PR Curve')
    plt.legend(bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    plt.grid()
    
    if save:
        plt.savefig(RESULTS_DIR+'/crossvalidation_pr_curve.png', bbox_inches='tight',dpi=dpi)   
    if show:
        plt.show()
    plt.close()

def plot_shap_summary(model=None, nsamples=50, X_train=None, X_test=None, feature_names=None, titlestr='', show=False, save=False, RESULTS_DIR=None, dpi=300):
    print("No. of samples used to build explainer and generate shap values: ", nsamples)
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, nsamples))
    X_test_sample = shap.sample(X_test, nsamples)
    shap_values = explainer.shap_values(X=X_test_sample)
    shap.summary_plot(shap_values=shap_values, features=X_test_sample, feature_names=feature_names, show=False)
    # shap.plots.violin(shap_values=shap_values, features=X_test, plot_type="layered_violin", show=False)
    plt.title(titlestr+'Shapley Analysis')
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR+'/shap_summary_plot.png', bbox_inches='tight',dpi=dpi)   
    if show:
        plt.show()

    plt.close()

def plot_calibration_curve(classifiers_list=[], 
                           classifiers_names=[],
                           X_train=None, X_test=None, y_train=None, y_test=None,
                           titlestr='', RESULTS_DIR='', show=False, save=False, 
                           dpi=300):
    print("Generating Calibration Curves...")
    probas_list = []
    for classifier, name in zip(classifiers_list, classifiers_names):
        print("\tTraining classifier: ", name)
        probas_list.append(classifier.fit(X_train, y_train).predict_proba(X_test))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax = scikitplot.metrics.plot_calibration_curve(y_test, probas_list, 
                                                    classifiers_names, 
                                                    text_fontsize='xx-large', 
                                                    ax=ax)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_facecolor('white')
    for l in ax.lines:
        l.set_lw(2)
    plt.title(titlestr+"Calibration Curve")
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR+'/calibration_curve.png', bbox_inches='tight',dpi=dpi)
    if show:
        plt.show()
    plt.close()

def eval_classification(make_metrics_plots=True, y_pred=None, y_pred_proba=None, 
                        untrained_model=None, n_splits=10, X=None, y=None, 
                        make_calibration_curves=False, untrained_models_list=[], untrained_models_names=[],
                        make_shap_plot=False, trained_model=None, shap_nsamples=50,
                        X_train=None, X_test=None, y_train=None, y_test=None,
                        class_names=None, feature_names=None,
                        titlestr="",
                        save=False, RESULTS_DIR=None,
                        show=False, dpi=300):

    classes_titlestr = ', '.join(["class {}: {}".format(i, cls_nm) for i, cls_nm in enumerate(class_names)])
    if titlestr != "":
        # titlestr_cls = titlestr+"\n("+classes_titlestr+")\n\n"
        # titlestr_nocls = titlestr+"\n\n"
        titlestr_cls = titlestr_nocls = titlestr+"\n"
    else:
        # titlestr_cls = "("+classes_titlestr+")\n\n"
        # titlestr_nocls = titlestr
        titlestr_cls = titlestr_nocls = ""

    if save:
        if RESULTS_DIR is not None:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            print("Saving results in {}".format(RESULTS_DIR))   
        else:
            print("Hey! You asked me to save results but did not provide a RESULTS_DIR !!!")

    if make_metrics_plots:
        if y_test is not None and y_pred is not None:
            assert len(y_test) == len(y_pred), "[Length Mismatch]: Number of test samples not equal to number of predictions"
            generate_classification_report(y_test=y_test, 
                                            y_pred=y_pred, 
                                            RESULTS_DIR=RESULTS_DIR, 
                                            labels=class_names,
                                            show=show, 
                                            save=save)

            plot_confusion_matrix(y_test=y_test, 
                                y_pred=y_pred, 
                                RESULTS_DIR=RESULTS_DIR, 
                                labels=class_names,
                                titlestr=titlestr_nocls,
                                dpi=dpi,
                                show=show,  
                                save=save)

        if y_test is not None and y_pred_proba is not None:
            assert len(y_test) == len(y_pred_proba), "[Length Mismatch]: Number of test samples not equal to number of predictions"
            
            if len(list(set(y_test))) == 2:
                plot_ks_stat(y_test=y_test,
                                y_pred=y_pred_proba, 
                                RESULTS_DIR=RESULTS_DIR, 
                                titlestr=titlestr_cls,
                                classes_titlestr = classes_titlestr,
                                dpi=dpi,
                                show=show, 
                                save=save)
                
                plot_lift_curve(y_test=y_test,
                                    y_pred=y_pred_proba, 
                                    RESULTS_DIR=RESULTS_DIR, 
                                    titlestr=titlestr_cls,
                                    classes_titlestr = classes_titlestr,
                                    dpi=dpi,
                                    show=show, 
                                    save=save)

                plot_cumul_gain(y_test=y_test,
                                    y_pred=y_pred_proba, 
                                    RESULTS_DIR=RESULTS_DIR, 
                                    titlestr=titlestr_cls,
                                    classes_titlestr = classes_titlestr,
                                    dpi=dpi,
                                    show=show, 
                                    save=save)
            else:
                print("Skipping KS, Lift and Cumulative Gain plots as number of classes is not 2")
            
            plot_classwise_pr_curve(y_test=y_test,
                                        y_pred=y_pred_proba, 
                                        RESULTS_DIR=RESULTS_DIR, 
                                        titlestr=titlestr_cls,
                                        classes_titlestr = classes_titlestr,
                                        dpi=dpi,
                                        show=show, 
                                        save=save)
            
            plot_classwise_roc_curve(y_test=y_test,
                                        y_pred=y_pred_proba, 
                                        RESULTS_DIR=RESULTS_DIR, 
                                        titlestr=titlestr_cls,
                                        classes_titlestr = classes_titlestr,
                                        dpi=dpi,
                                        show=show, 
                                        save=save)

        if untrained_model is not None and X is not None and y is not None:
            assert len(X) == len(y), "[Length Mismatch]: Number of samples not equal to number of class labels"    
            if len(list(set(y_test))) == 2:
                
                plot_cv_roc_curve(classifier=untrained_model, 
                                            X=X, 
                                            y=y, 
                                            n_splits=n_splits,
                                            RESULTS_DIR=RESULTS_DIR, 
                                            titlestr=titlestr_nocls, 
                                            dpi=dpi,
                                            show=show, 
                                            save=save)
                
                plot_cv_pr_curve(classifier=untrained_model,  
                                    X=X, 
                                    y=y, 
                                    n_splits=n_splits,
                                    RESULTS_DIR=RESULTS_DIR, 
                                    titlestr=titlestr_nocls,
                                    dpi=dpi,
                                    show=show, 
                                    save=save)
            else:
                print("Skipping Cross Validated ROC and PR curves as number of classes is not 2")
    
    if make_shap_plot:
        if X_train is not None and X_test is not None and trained_model is not None:
            plot_shap_summary(model=trained_model, X_train=X_train, X_test=X_test,
                                nsamples=shap_nsamples, feature_names=feature_names,
                                titlestr=titlestr_nocls, show=show, RESULTS_DIR=RESULTS_DIR, save=save)
        else:
            print("Hey! You asked me to make a shap plot but did not provide a trained model, X_train and X_test !!!")
    
    if make_calibration_curves:
        if len(list(set(y_test))) == 2:
            if untrained_models_list is not None and \
                untrained_models_names is not None and \
                    X_train is not None and y_train is not None and \
                        X_test is not None and y_train is not None:
                assert len(untrained_models_list) == len(untrained_models_names), "[Length Mismatch]: Number of models not equal to number of model names"
                assert len(X_train) == len(y_train), "[Length Mismatch]: Number of samples not equal to number of class labels"
                assert len(X_test) == len(y_test), "[Length Mismatch]: Number of samples not equal to number of class labels"

                plot_calibration_curve(classifiers_list=untrained_models_list,
                                        classifiers_names=untrained_models_names,
                                        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                        titlestr=titlestr_nocls, RESULTS_DIR=RESULTS_DIR, show=show, save=save, dpi=dpi)
            else:
                print("Skipping Calibration Curves as number of classes is not >=2 ")

            