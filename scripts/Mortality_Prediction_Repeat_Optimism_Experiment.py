# Cross Validation - In-hospital Mortality Prediction (Optimism Error)
#####-----------------------------------------------------------------

# import libraries
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# load processed input data
data = pd.read_csv('../data/MIMIC3_FULL_PROCESSED_DATASET_20221022.csv')
data.head()
data.shape
data.SUBJECT_ID.nunique()
data.HADM_ID.nunique()
data.ICUSTAY_ID.nunique()
print(list(data.columns))
data.isna().sum().sort_values(ascending=False)
data.dtypes

# get selected features for mortality prediction
race_cols = [x for x in data.columns if 'RACE' in x]
data['GENDER_FEMALE'] = data.GENDER.apply(lambda x: 1 if x=='F' else 0)
dem_cols = ['GENDER_MALE', 'GENDER_FEMALE', 'AGE'] + race_cols
drop_cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ADMITTIME', 'DISCHTIME', 'ETHNICITY', 'GENDER']
label_cols = [x for x in data.columns if 'LABEL' in x]

X = data.drop(drop_cols + label_cols, axis=1)
X.shape
y = data['MORTALITY_INHOSPITAL_STAY_LABEL']
y.shape
np.any(X.isna())
np.mean(y)
y.value_counts()

# defining column types for pre-processing
binary_dem_cols = ['GENDER_MALE', 'GENDER_FEMALE', 'RACE_WHITE', 'RACE_HISPANIC', 'RACE_BLACK',
                   'RACE_ASIAN', 'RACE_OTHER_UNKNOWN']
binary_diag_cols = ['Septicemia (except in labor)', 'Diabetes mellitus without complication',
                    'Diabetes mellitus with complications', 'Disorders of lipid metabolism',
                    'Fluid and electrolyte disorders', 'Essential hypertension',
                    'Hypertension with complications and secondary hypertension', 'Acute myocardial infarction',
                    'Coronary atherosclerosis and other heart disease', 'Conduction disorders', 'Cardiac dysrhythmias',
                    'Congestive heart failure; nonhypertensive', 'Acute cerebrovascular disease',
                    'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
                    'Chronic obstructive pulmonary disease and bronchiectasis', 'Pleurisy; pneumothorax; pulmonary collapse',
                    'Respiratory failure; insufficiency; arrest (adult)', 'Other lower respiratory disease',
                    'Other upper respiratory disease', 'Other liver diseases', 'Gastrointestinal hemorrhage',
                    'Acute and unspecified renal failure', 'Chronic kidney disease',
                    'Complications of surgical procedures or medical care', 'Shock']
binary_admit_cols = ['ADMIT_DIAG_CARDIAC_ARREST', 'ADMIT_DIAG_BRAIN_HEMORRHAGE', 'ADMIT_DIAG_LIVER FAILURE',
                     'ADMIT_DIAG_CEREBROVASCULAR ACCIDENT', 'ADMIT_DIAG_SEPSIS', 'ADMIT_DIAG_HYPOXIA',
                     'ADMIT_DIAG_RESPIRATORY DISTRESS']
dem_cont_cols = ['AGE', 'HEIGHT', 'WEIGHT']
meas_cont_cols = ['Capillary refill rate', 'Diastolic blood pressure',
                  'Fraction inspired oxygen', 'Glascow coma scale total', 'Glucose', 'Heart Rate',
                  'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
                  'Systolic blood pressure', 'Temperature', 'pH']

# rename sets of feature columns
X.columns = [x.upper() for x in X.columns if x not in meas_cont_cols]
binary_dem_cols = [x.upper() for x in binary_dem_cols]
binary_diag_cols = [x.upper() for x in binary_diag_cols]
binary_admit_cols = [x.upper() for x in binary_admit_cols]
dem_cont_cols = [x.upper() for x in dem_cont_cols]

# describe feature ranges
col_sets = [binary_dem_cols, binary_diag_cols, binary_admit_cols, dem_cont_cols]
names = ['Demographics', 'CCS Diagnoses', 'Primary Admission Reason', 'Measurement Variables']

# set outlier age/weight/height values to NA for imputation
X.loc[(X.AGE > 110), 'AGE'] = np.nan
X.loc[(X.WEIGHT == 0), 'WEIGHT'] = np.nan
X.loc[(X.HEIGHT==0), 'HEIGHT'] = np.nan
cont_cols = dem_cont_cols
binary_cols = binary_dem_cols + binary_diag_cols + binary_admit_cols
features = cont_cols + binary_cols

## Updated Optimism Experiment: Estimate Test Error from CV Performances

# - Define 20% validation set and run CV on remaining 80% <br>
# - Run CV with different methods and return best estimators <br>
# - Predict on validation set with best CV estimator and compare results to within-CV metrics

# **Analyses: Repeat Validation-CV Optimism Comparison over only 5-Folds**

# import additional libraries used by functions below
import scipy.stats as st
import joblib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, LeavePOut
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, TimeSeriesSplit
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.feature_selection import RFECV, SelectKBest, SelectFdr, chi2, mutual_info_classif, RFE, SelectFromModel
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score, label_ranking_loss, zero_one_loss
from sklearn.impute import SimpleImputer, KNNImputer


# define function for nested cross-validation with all model selection steps
def nested_cv(X, y, model='LogisticRegression', cpu_num=-1, n_folds=5, verbose=True,
              return_predictions=False, return_models=True, feature_selection=True, scaling=True,
              cont_cols=None, binary_cols=None, random_state=8888): # add imputation methods/Fselection options
    "Perform nested cross validation and get outcomes & predictions"
    
    # ----- NOTE: NEED TO FILL IN WITH YOUR FEATURE NAMES BELOW -----
    # define column transformers based on given feature subsets
    ss_cols = cont_cols # to use with standard scaler

    # combine transformers with Column Transformer
    ss = StandardScaler()

    overall_transformer = ColumnTransformer(
        transformers=[
            ("standard_scaler", ss, ss_cols),
        ]
    )
    
    imputer = SimpleImputer(missing_values=np.nan, fill_value='median')
    feature_selector = SelectKBest(mutual_info_classif)
    
    # specify classifier to use
    if model == 'LogisticRegression':
        clf = LogisticRegression(solver='saga', n_jobs=-1, random_state=8888)
    elif model == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=8888)
    else:
        print('Error: Must specify one of the acceptable classifiers')
        return

    # pipeline requires ordered input for preprocessing
    pipe = Pipeline(steps=[('imputer', imputer),
                           ('preprocessor', ss),
                           ('select', feature_selector),
                           ('classify', clf)])

    # double underscore allows access to pipeline step
    pipe_params = {'classify__C': np.power(10., np.arange(-2,2)),
                   'classify__penalty': ['l1', 'l2', 'none'],
                   'select__k': [15, 30]}

    # reset indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # initialize dictionary to hold CV results and classifiers
    start = time()
    result_dict = {}
    overall_preds = np.zeros(len(X))
    overall_clfs = []
    result_dict['estimator'] = []
    result_dict['test_roc_auc'] = []
    result_dict['test_average_precision'] = []
    
    # begin outer fold splitting
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    i = 1
    for train_index, test_index in skf.split(X, y):
        if verbose:
            print('OUTER FOLD ' + str(i) + ':')
        X_train = X.iloc[train_index, :]
        y_train = y[train_index]
        X_test = X.iloc[test_index, :]
        y_test = y[test_index]
        
        # run cv using training set from each outer fold split
        gscv = GridSearchCV(pipe, pipe_params, cv=n_folds, verbose=verbose, n_jobs=cpu_num,
                            scoring='average_precision', return_train_score=False, refit=True)
        gscv = gscv.fit(X_train, y_train)
        
        # get best classifier and predict on outer test set
        #cv_models = cv['estimator']
        #best_clf = cv_models[np.argmax(cv['test_average_precision'])]
        best_clf = gscv.best_estimator_
        y_pred = best_clf.predict_proba(X_test)[:, 1]
        result_dict['test_average_precision'].append(average_precision_score(y_true = y_test, y_score=y_pred))
        result_dict['test_roc_auc'].append(roc_auc_score(y_true = y_test, y_score=y_pred))
        result_dict['estimator'].append(best_clf)
        i += 1
        
    # print total time required
    total_time = time() - start
    if verbose:
        print('\nTOTAL RUNTIME (s): {}'.format(total_time))
        
    result_dict['total_runtime'] = total_time
    return result_dict
#     # get final preds and save models
#     if save_models:
#         for k, model_obj in enumerate(overall_clfs):
#             joblib.dump(model_obj, model_path  + subset_name + '_' + 'LassoRegression' + '_CV_' + str(k) + '.pkl')
        
#     # build result df
#     result_df = data[["PERSON_ID", "GRID", "OUTCOME"]]
#     result_df['PREDS'] = overall_preds
    
#     # optionally save results to csv
#     if save_final_results:
#         file_name = subset_name + '_' + 'LassoRegression' + '_EARLY_FUSION_PREDICTIONS.csv'
#         result_df.to_csv(preds_path + file_name, index=False)
    
#     # optionally return results
#     if return_results:
#         return result_df

# define function to apply non-nested CV experiments with optional parameters
def apply_cross_validation(X, y, model='LogisticRegression', n_folds=5, n_repeats=10,
                           verbose=True):
    # define methods to use for cross validation
    kf = KFold(n_splits=n_folds, shuffle=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    repeated_kf = RepeatedKFold(n_repeats=n_repeats, n_splits=n_folds)
    repeated_skf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_folds)
    cv_methods = [kf, skf, repeated_kf, repeated_skf]
    cv_names = ['KFoldCV', 'StratifiedCV', 'RepeatedCV', 'RepeatedStratifiedCV']
    cv_types = dict(zip(cv_names, cv_methods))
    
    # define column transformers based on given feature subsets
    ss_cols = cont_cols # to use with standard scaler

    # combine transformers with Column Transformer
    ss = StandardScaler()

    overall_transformer = ColumnTransformer(
        transformers=[
            ("standard_scaler", ss, ss_cols),
        ]
    )

    #imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')

    #feature_selector = SelectFdr(mutual_info_classif)
    #feature_selector = SelectFromModel(LogisticRegression(n_jobs=-1))
    #feature_selector = RFE(mutual_info_classif)
    feature_selector = SelectKBest(mutual_info_classif)

    # specify classifier to use
    if model == 'LogisticRegression':
        clf = LogisticRegression(solver='saga', n_jobs=-1)
    elif model == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    else:
        print('Must specify one of the allowed classifiers')
        pass

    # pipeline requires ordered input for preprocessing
    pipe = Pipeline(steps=[('imputer', imputer),
                           ('preprocessor', ss),
                           ('select', feature_selector),
                           ('classify', clf)])

    # double underscore allows access to pipeline step
    pipe_params = {'classify__C': np.power(10., np.arange(-2,2)),
                   'classify__penalty': ['l1', 'l2', 'none'],
                   'select__k': [15, 30]}
    
    
    # initalize objects to hold cv results
    cv_dict = {}
    cv_dfs = []
    cv_run_times = []
    cv_types = dict(zip(cv_names, cv_methods))
    
    # iterate over each CV method and apply grid search
    for name, split_fn in cv_types.items():
        print(name)
        start_time = time()
        gs = GridSearchCV(pipe, pipe_params, cv=split_fn, verbose=True, n_jobs=-1,
                          scoring=['average_precision', 'roc_auc'],
                          return_train_score=False, refit="average_precision")
        gs.fit(X, y)
        cv = gs.cv_results_
        total_time = time() - start_time
        cv['total_runtime'] = total_time
        cv_run_times.append(total_time)
        df = pd.DataFrame(cv)
        df['method'] = name
        cv_dfs.append(df)
        cv['gs_best_score_'] = gs.best_score_
        cv['gs_best_estimator_'] = gs.best_estimator_
        cv_dict[name] = cv
    return cv_dict

# function for comparing CV and validation set metrics
def compare_cv_validation_metrics(cv_dict, X_val, y_val):
    val_dict = {}
    val_auprs = []
    val_aurocs = []
    for cv_method in list(cv_dict.keys()):
        run_time = cv_dict[cv_method]['total_runtime']
        cv_auprs = cv_dict[cv_method]['best_auprs']
        cv_aurocs = cv_dict[cv_method]['best_aurocs']
        best_mean_aupr = np.mean(cv_auprs)
        best_mean_auroc = np.mean(cv_aurocs)
        best_aupr = np.max(cv_auprs)
        best_auroc = np.max(cv_aurocs)
        best_clf = cv_dict[cv_method]['best_estimator']

        # predict on validation set using best model from CV
        y_pred = best_clf.predict_proba(X_val)[:, 1]
        val_aupr = average_precision_score(y_true = y_val, y_score=y_pred)
        val_auroc = roc_auc_score(y_true = y_val, y_score=y_pred)
        val_auprs.append(val_aupr)
        val_aurocs.append(val_auroc)

        # get metrics and save as dict
        val_dict[cv_method] = {}
        val_dict[cv_method]['Runtime'] = run_time
        val_dict[cv_method]['CV_Best_Avg_AUPR'] = best_mean_aupr
        val_dict[cv_method]['CV_Best_Avg_AUROC'] = best_mean_auroc
        val_dict[cv_method]['CV_Best_AUPR'] = best_aupr
        val_dict[cv_method]['CV_Best_AUROC'] = best_auroc
        val_dict[cv_method]['Val_AUPR'] = val_aupr
        val_dict[cv_method]['Val_AUROC'] = val_auroc
        val_dict[cv_method]['Val_Relative_Best_CV_AUPR'] =  val_aupr / best_aupr
        val_dict[cv_method]['Val_Relative_Best_CV_AUROC'] = val_auroc / best_auroc
        val_dict[cv_method]['Val_Relative_Avg_CV_AUPR'] =  val_aupr / best_mean_aupr
        val_dict[cv_method]['Val_Relative_Avg_CV_AUROC'] = val_auroc / best_mean_auroc
    
    # return final metrics as dictionary
    return val_dict

# Load the dataset defined above
X_exp1 = X[features].copy()
y_exp1 = y.copy()

# split data into separate validation set (use to estimate true test error)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_exp1, y_exp1, random_state=8888, shuffle=True,
                                                  stratify=y_exp1, test_size=0.2)

# examine training dataset (to use with CV) and validation dataset
print('Examining inner train set to use with CV')
X_train.head()
X_train.shape
y_train.head()
y_train.shape
y_train.mean()

print('Examining held-out validation set for CV test error')
X_val.head()
X_val.shape
y_val.head()
y_val.shape
y_val.mean()

# define methods to use for cross validation
kf = KFold(n_splits=5, shuffle=True)
skf = StratifiedKFold(n_splits=5, shuffle=True)
repeated_kf = RepeatedKFold(n_repeats=10, n_splits=5)
repeated_skf = RepeatedStratifiedKFold(n_repeats=10, n_splits=5)

cv_methods = [kf, skf, repeated_kf, repeated_skf]
cv_names = ['KFoldCV', 'StratifiedCV', 'RepeatedCV', 'RepeatedStratifiedCV']
cv_types = dict(zip(cv_names, cv_methods))

# repeat non-nested CV & validation set comparison multiple times
print('Begin Non-nested CV & Validation Set Comparison\n')

NUM_TRIALS = 10
val_result_dfs = []
for i in range(NUM_TRIALS):
    print('Trial #: {}'.format(i+1))
    
    # get train/test sets to use for CV & validation set metrics (don't set same random seed)
    X_train, X_val, y_train, y_val = train_test_split(X_exp1, y_exp1, shuffle=True, stratify=y_exp1,
                                                      test_size=0.2)
    
    # run experiment applying all non-nested CV methods
    cv_dict = apply_cross_validation(X_train, y_train, n_folds=5)
    
    # store performance/runtime results
    auc_dict = {}
    for cv_method in cv_names:
        if 'Repeated' in cv_method:
            n_splits = 50
        else:
            n_splits = 5
        auc_dict[cv_method] = {}
        
        # get aupr/auroc scores from best grid search configuration model
        best_aupr_index = np.argmin(cv_dict[cv_method]['rank_test_average_precision'])
        best_auprs = []
        best_auroc_index = np.argmin(cv_dict[cv_method]['rank_test_roc_auc'])
        best_aurocs = []
        
        # store best results for each split
        for k in range(n_splits):
            aupr = cv_dict[cv_method]['split' + str(k) + '_test_average_precision'][best_aupr_index]
            best_auprs.append(aupr)
            auroc = cv_dict[cv_method]['split' + str(k) + '_test_roc_auc'][best_auroc_index]
            best_aurocs.append(auroc)
            
        # store overall results and pass to validation comparison function
        mean_aupr = np.mean(best_auprs)
        std_aupr = np.std(best_auprs)
        mean_auroc = np.mean(best_aurocs)
        std_auroc = np.std(best_aurocs)
        auc_dict[cv_method]['mean_aupr'] = round(mean_aupr, 3)
        auc_dict[cv_method]['std_aupr'] = round(std_aupr, 3)
        auc_dict[cv_method]['best_auprs'] = best_auprs
        auc_dict[cv_method]['mean_auroc'] = round(mean_auroc, 3)
        auc_dict[cv_method]['std_auroc'] = round(std_auroc, 3)
        auc_dict[cv_method]['best_aurocs'] = best_aurocs
        auc_dict[cv_method]['best_estimator'] = cv_dict[cv_method]['gs_best_estimator_']
        auc_dict[cv_method]['total_runtime'] = cv_dict[cv_method]['total_runtime']
    
    # compare cross-validation performance with initially held-out validation set 
    val_metrics = compare_cv_validation_metrics(auc_dict, X_val, y_val)
    val_df = pd.DataFrame(val_metrics)
    val_df['trial_num'] = i + 1
    print('Trial {} Finished - Saving results to csv'.format(str(i+1)))
    val_df.to_csv('../plots/MORTALITY_PREDICTION_CV_VALIDATION_REPEAT_TRIAL_' + str(i+1) + '_5_FOLDS_WITH_TUNING.csv', index=False)
    val_result_dfs.append(val_df)

# examine results from non-nested methods and write result to csv
print('Examining combined non-nested optimism data frame:')
val_result_df = pd.concat(val_result_dfs)
#val_result_df = val_result_df.reset_index().rename(columns=({'index':'Metric'}))
val_result_df.head()
val_result_df.shape
#val_result_df.to_csv('../plots/MORTALITY_PREDICTION_CV_VALIDATION_REPEATED_5_FOLDS_WITH_TUNING.csv', index=False)

# repeat nested CV & validation set comparison over several trials
print('Begin nested CV & Validation Set Comparison\n')
NUM_TRIALS = 10
nested_cv_result_dfs = []
nested_cv_result_dicts = []
nested_cv_val_dfs = []
for i in range(NUM_TRIALS):
    print('Trial #: {}'.format(i+1))
    X_train, X_val, y_train, y_val = train_test_split(X_exp1, y_exp1, shuffle=True, stratify=y_exp1,
                                                      test_size=0.2)
    nested_cv_results = nested_cv(X_train, y_train, n_folds=5,
                                  cont_cols=cont_cols, binary_cols=binary_cols)
    nested_cv_result_dicts.append(nested_cv_results)
    nested_cv_result_df = pd.DataFrame(nested_cv_results)
    nested_cv_result_df['trial_num'] = i+1
    nested_cv_result_dfs.append(pd.DataFrame(nested_cv_result_df))
    
    nested_mean_aupr = np.mean(nested_cv_results['test_average_precision'])
    nested_mean_auroc = np.mean(nested_cv_results['test_roc_auc'])
    nested_std_aupr = np.std(nested_cv_results['test_average_precision'])
    nested_std_auroc = np.std(nested_cv_results['test_roc_auc'])
    
    nested_cv_dict = {}
    nested_cv_dict['NestedCV'] = {'mean_aupr': round(nested_mean_aupr, 3),
                                  'std_aupr': round(nested_std_aupr, 3),
                                  'mean_auroc': round(nested_mean_auroc, 3),
                                  'std_auroc': round(nested_std_auroc, 3),
                                  'best_auprs': nested_cv_results['test_average_precision'],
                                  'best_aurocs': nested_cv_results['test_roc_auc'],
                                  'best_estimator': nested_cv_results['estimator'][np.argmax(nested_cv_results['test_average_precision'])], 
                                  'total_runtime': nested_cv_results['total_runtime']}
    
    nested_cv_val_metrics = compare_cv_validation_metrics(nested_cv_dict, X_val, y_val)
    nested_cv_val_df = pd.DataFrame(nested_cv_val_metrics)
    nested_cv_val_df['trial_num'] = i+1
    nested_cv_val_dfs.append(nested_cv_val_df)
    print('Trial {} Finished - Saving results to csv'.format(str(i+1)))
    nested_cv_val_df.to_csv('../plots/MORTALITY_PREDICTION_NESTED_CV_VALIDATION_REPEAT_TRIAL_' + str(i+1) + '_5_FOLDS_WITH_TUNING.csv', index=False)

# examine results from nested cv and write result to csv
print('Examining combined nested optimism data frame:')
nested_result_df = pd.concat(nested_cv_val_dfs)
#val_result_df = val_result_df.reset_index().rename(columns=({'index':'Metric'}))
nested_result_df.head()
nested_result_df.shape
#nested_cv_val_results.to_csv('../plots/MORTALITY_PREDICTION_NESTED_CV_VALIDATION_REPEATED_5_FOLDS_WITH_TUNING.csv', index=False)

# exit script
print('Optimism Experiment Complete. Exiting process')
