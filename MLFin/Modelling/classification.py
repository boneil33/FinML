import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV, LogisticRegression, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, f1_score, make_scorer, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.base import clone


def run_cv(raw, ct, feat_adj, is_start_dt, num_features, target, sample_weight=None, 
           tss_splits=20, model=Lasso(), model_type='Regression', max_train_size=None, 
           hyperparam_name='alpha', param_space=np.logspace(-4,4,50), scorer=None, cat_features=[]):
    """
        Run a simple cross validation on lasso to determine whether any of these features 
        have predictive power.
    
    :param r: (dict of dataframes) output from get_rolling_cts, full sample we'll chop test off 0.25
    :param ct: (string) contract to look at
    :param feat_adj: (string) adjuster for features (e.g. _dv01)
    :param sample_weight: (string) colname for sample_weight
    :return: (GridSearchCV, pd.Series) the search object and info from Ridge cross validated on contract data
        betas: array
        best_score: neg MSE
        best_score_ratio: neg MSE/null prior(constant, average prediction)
        alpha: best ridge alpha
        
    """
    
    data = raw.copy(deep=True).loc[is_start_dt:].dropna(how='any')
    if sample_weight is None:
        sample_weight_arr = []
    else:
        sample_weight_arr= [sample_weight]
    features = np.hstack([num_features, cat_features, sample_weight_arr])
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    is_end_dt = y_test.index[-1]

    # standardize variables, going to use full sample normalization don't think it'll matter much 
    #    since the data is already fairly stationary
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler(with_mean=True))
    ])
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    #combine different feature types
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    tss = TimeSeriesSplit(n_splits=tss_splits, max_train_size=max_train_size)
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    if not sample_weight :
        sample_weight_data = None
    else:
        sample_weight_data = X_train[sample_weight]
        X_train = X_train.drop(sample_weight, axis=1)
        
    search = cv_gs_wrap(pipe, X_train, y_train, hyperparam_name, param_space, tss_splits=tss_splits, 
                        sample_weight=sample_weight_data, scorer=scorer, max_train_size=max_train_size)
    
    if model_type.lower()=='regression':
        dummy_clf = DummyRegressor(strategy='mean') # will just guess the average
    elif model_type.lower()=='classification':
        dummy_clf = DummyClassifier(strategy='prior') # will guess the most frequent class
    else:
        raise ValueError(r"model_type {0} not accepted must be Regression or Classification".format(model_type))
    dummy_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('dummy', dummy_clf)
    ]).fit(X_train, y_train)
    
    dummy_score = cross_val_score(dummy_pipe, X_train, y_train, cv=tss, scoring=scorer)
    output_df = pd.Series(name=ct)
    output_df['beta'] = search.best_estimator_[-1].coef_
    output_df['best_score'] = search.best_score_
    output_df[hyperparam_name] = search.best_params_['model__'+hyperparam_name]
    output_df['best_score_ratio'] = search.best_score_ / dummy_score.mean()
    
    return search, output_df


# cross validate to find regularization parameter C = 1/lambda
def cv_gs_wrap(clf, X_train, y_train, reg_param_name, param_space, sample_weight=None, 
               tss_splits=5, scorer='f1', verbose=False, max_train_size=None):
    """
    Wrapper for gridsearchcv with weighting
    Doesn't appear you can fit with sample weights and score with them, either one or the other and the latter is a hack
    if sample_weight is not none, scorer needs to be a make_scorer that takes sample weights (e.g. score_f1_weighted below)
    """
    param_grid = {'model__'+reg_param_name: param_space}#np.logspace(-4, 2, 50)}
    fit_params = {'model__sample_weight': sample_weight}
    tss = TimeSeriesSplit(n_splits=tss_splits, max_train_size=max_train_size)
    
    
    if sample_weight is not None:
        search = GridSearchCV(clf, param_grid, scoring=scorer, n_jobs=-1, cv=tss)
        search.fit(X_train, y_train, **fit_params)
    else:
        search = GridSearchCV(clf, param_grid, scoring=scorer, n_jobs=-1, cv=tss)
        search.fit(X_train, y_train)
    if verbose:
        print(search.best_score_)
        print(search.best_params_)

    return search


# cross valididation with weights passed to scoring
def cv_gs_wrap_weighted(base_clf, X, y, reg_param_name, param_space, sample_weight, 
               tss_splits=5, scorer='accuracy', verbose=False, max_train_size=None, clf_params={}):
    """
        In order to apply sample weights to the scorer we need to implement our own gridsearchcv
    :param base_clf: a classifier that implements predict, we will apply params later
    """
    if scorer not in ['accuracy', 'f1'] or sample_weight is None:
        raise ValueError('only implemented for mean accuracy scoring with weights')
    
    
    tss = TimeSeriesSplit(n_splits=tss_splits, max_train_size=max_train_size)
    #search = GridSearchCV(clf, param_grid, scoring=scorer, n_jobs=-1, cv=tss)
    #search.fit(X_train, y_train, **fit_params)
    output_srs = pd.Series([[], 0, None], index=['beta', 'score', reg_param_name])
    for hyperparam in param_space:
        clf_params['model__'+reg_param_name] = hyperparam
        clf = clone(base_clf)
        clf.set_params(**clf_params)
        scores = []
        for train_idx, test_idx in tss.split(X=X, y=y):
            X_train = X.iloc[train_idx, :]
            y_train = y.iloc[train_idx]
            w_train = sample_weight.iloc[train_idx]
            X_test = X.iloc[test_idx, :]
            y_test = y.iloc[test_idx]
            w_test = sample_weight.iloc[test_idx]
            
            # fit, predict and score with weights
            clf.fit(X_train, y_train, model__sample_weight=w_train)
            y_pred = clf.predict(X=X_test)
            if scorer=='accuracy':
                score = accuracy_score(y_test, y_pred, sample_weight=w_test)
            elif scorer=='f1':
                # props: https://stackoverflow.com/questions/49581104/sklearn-gridsearchcv-not-using-sample-weight-in-score-function
                score = f1_score(y_test, y_pred, sample_weight=w_test)
                
            scores.append(score)
        avg_score = np.array(scores).mean()
        if avg_score > output_srs['score']:
            clf.fit(X, y, model__sample_weight=sample_weight)
            if hasattr(clf.named_steps['model'], 'coef_'):
                beta = clf.named_steps['model'].coef_
            else:
                beta = None
            output_srs['beta'] = beta
            output_srs['score'] = avg_score
            output_srs[reg_param_name] = hyperparam
        
    # set best param and refit
    clf_params['model__'+reg_param_name] = output_srs[reg_param_name]
    ret_clf = clone(base_clf)
    ret_clf.set_params(**clf_params)
    ret_clf.fit(X, y, model__sample_weight=sample_weight)
        
    return output_srs, ret_clf
    

def run_cv_weighted(raw, ct, feat_adj, is_start_dt, num_features, target, sample_weight_name, tss_splits=20, 
                    model=LogisticRegression(multi_class='auto'), max_train_size=None, hyperparam_name='C',
                    param_space=np.logspace(-4,4,50), scorer='accuracy', cat_features=[], clf_params={}, return_clf=False):
    """
        Run a simple cross validation on lasso to determine whether any of these features 
        have predictive power with scores weighted.
    
    :param r: (dict of dataframes) output from get_rolling_cts, full sample we'll chop test off 0.25
    :param ct: (string) contract to look at
    :param feat_adj: (string) adjuster for features (e.g. _dv01)
    :param sample_weight_name: (string) colname for sample_weight
    :return: (GridSearchCV, pd.Series) the search object and info from Ridge cross validated on contract data
        betas: array
        best_score: neg MSE
        best_score_ratio: neg MSE/null prior(constant, average prediction)
        alpha: best ridge alpha
        
    """
    
    data = raw.copy(deep=True).loc[is_start_dt:].dropna(how='any')
    if sample_weight_name is None:
        sample_weight_arr = []
    else:
        sample_weight_arr= [sample_weight_name]
    features = np.hstack([num_features, cat_features, sample_weight_arr])
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    # standardize variables, going to use full sample normalization don't think it'll matter much 
    #    since the data is already fairly stationary
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler(with_mean=True))
    ])
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    #combine different feature types
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    tss = TimeSeriesSplit(n_splits=tss_splits, max_train_size=max_train_size)
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    sample_weight_data = X_train[sample_weight_name]
    X_train = X_train.drop(sample_weight_name, axis=1)
        
    output_srs, output_clf = cv_gs_wrap_weighted(pipe, X_train, y_train, hyperparam_name, param_space,
                                     sample_weight_data, tss_splits=tss_splits, scorer=scorer, 
                                     max_train_size=max_train_size, clf_params=clf_params)
    
    
    dummy_clf = DummyClassifier(strategy='prior') # will guess the most frequent class
    
    dummy_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('dummy', dummy_clf)
    ]).fit(X_train, y_train)
    
    dummy_score = cross_val_score(dummy_pipe, X_train, y_train, cv=tss, scoring=scorer)
    output_df = pd.Series(name=ct)
    output_df['beta'] = output_srs['beta']
    output_df['best_score'] = output_srs['score']
    output_df[hyperparam_name] = output_srs[hyperparam_name]
    output_df['best_score_ratio'] = output_srs['score'] / dummy_score.mean()
    
    if return_clf:
        return output_df, output_clf
    else:
        return output_df


def score_f1_weighted(y_true, y_pred, sample_weight):
    """
    props: https://stackoverflow.com/questions/49581104/sklearn-gridsearchcv-not-using-sample-weight-in-score-function
    """
    print(sample_weight,1+'s')
    return f1_score(y_true.values, y_pred, sample_weight=sample_weight.loc[y_true.index.values].values.reshape(-1))


# display CV metrics
def display_cv_metrics(search, reg_param_name, scorer, log_scale=False):
    evalScore = np.array(search.cv_results_['mean_test_score'])
    evalC = np.array(search.cv_results_['param_model__'+reg_param_name].data)
    scorer_name = ''
    if isinstance(scorer, str):
        scorer_name = scorer
    fig, ax = plt.subplots()
    ax.plot(evalC, evalScore)
    ax.set_ylabel(scorer_name)
    ax.set_xlabel(reg_param_name)
    if log_scale:
        ax.set_xscale('log')
    ax.set_title(scorer_name +' vs. '+reg_param_name+' (regularization param)')
    plt.show()
    
    
# look at some evaluation metrics of the training set with the optimal regularization parameter
def display_is_metrics(search, X_train, y_train, y_prob=None, y_pred=None):
    if search is None and y_prob is None:
        raise ValueError('Need either search or y_prob/y_pred')
    if search:
        if hasattr(search, 'decision_function'):
            y_prob = search.decision_function(X_train)
        else:
            y_prob = search.predict_proba(X_train)[:,1]
        y_pred = search.predict(X_train)
    precision, recall, thresh = precision_recall_curve(y_train, y_prob)
    fp_rate, tp_rate, _ = roc_curve(y_train, y_prob)
    fig, ax = plt.subplots(figsize=(8,6),nrows=2)
    ax[0].plot(recall, precision)
    ax[0].set_ylabel('precision')
    ax[0].set_xlabel('recall')
    ax[1].plot(fp_rate, tp_rate)
    ax[1].set_ylabel('true pos')
    ax[1].set_xlabel('false pos')
    fig.tight_layout()
    fig.suptitle('In-Sample CV Metrics', y=1.05)
    plt.show()
    cm = confusion_matrix(y_train, y_pred)
    print(pd.DataFrame(cm, index=['True_0','True_1'], columns=['Pred_0','Pred_1']))
    
    
# look at some evaluation metrics of the test set with the optimal regularization parameter
def display_oos_metrics(search, X_test, y_test, y_prob_t=None, y_pred_t=None):
    if search is None and y_prob_t is None:
        raise ValueError('Need either search or y_prob/y_pred')
    if search:
        if hasattr(search, 'decision_function'):
                y_prob_t = search.decision_function(X_test)
        else:
            y_prob_t = search.predict_proba(X_test)[:,1]
        y_pred_t = search.predict(X_test)
    precision, recall, thresh = precision_recall_curve(y_test, y_prob_t)
    fp_rate, tp_rate, _ = roc_curve(y_test, y_prob_t)
    fig, ax = plt.subplots(figsize=(8,6),nrows=2)
    ax[0].plot(recall, precision)
    ax[0].set_ylabel('precision')
    ax[0].set_xlabel('recall')
    ax[1].plot(fp_rate, tp_rate)
    ax[1].set_ylabel('true pos')
    ax[1].set_xlabel('false pos')
    ax[1].plot([0,1],[0,1], linestyle='--', color='r', alpha=.8)
    fig.tight_layout()
    fig.suptitle('Test Metrics', y=1.05)
    plt.show()
    cm = confusion_matrix(y_test, y_pred_t)
    print(pd.DataFrame(cm, index=['True_0','True_1'], columns=['Pred_0','Pred_1']))
    
    
# for logistic regression, just the model coefficients best we can do- using l2 normalization so should shrink losers to 0
def display_feature_imp(search, num_features):
    if isinstance(search, GridSearchCV):
        best_clf = search.best_estimator_.named_steps['model']
    else:
        best_clf = search
        
    if hasattr(best_clf, 'feature_importances_'):
        feature_imp = best_clf.feature_importances_
        sorted_idx = feature_imp.argsort()
        y_tick = np.arange(0, len(num_features))
        fig, ax = plt.subplots()
        ax.barh(y_tick, feature_imp[sorted_idx])
        ax.set_yticklabels(np.array(num_features)[sorted_idx])
        ax.set_yticks(y_tick)
        ax.set_title("RF Feature Importances (MDI)")
    else: #logisticclassifier
        best_coefs = best_clf.coef_[0]

        # get pesky feature names from onehotencoder
        clf_features = num_features.copy()
        # clf_features.extend(best_clf.named_steps['preprocessor'].transformers_[1][1].\
        #                     named_steps['onehot'].get_feature_names(cat_features))

        # sort them and barchart
        best_coefs = pd.Series(best_coefs, index=clf_features)
        best_coefs.sort_values(inplace=True)
        best_coefs = 100*(best_coefs/abs(best_coefs).max())
        fig, ax = plt.subplots()
        ytick = np.arange(best_coefs.shape[0])
        ax.barh(ytick, best_coefs, align='center')
        ax.set_yticks(ytick)
        ax.set_yticklabels(best_coefs.index)
        ax.set_title('abs coef size % max coef')
        
    fig.tight_layout()
    plt.show()


if __name__=='__main__':
  # which are numeric and which are categorical
  num_features = []###
  cat_features = []#
  pred_feature = ''
  pred_weight = ''

  # preprocess pipes
  num_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median')),
      ('scale', StandardScaler(with_mean=True))])

  cat_transformer = Pipeline(steps=[
      ('onehot', OneHotEncoder(handle_unknown='ignore'))]) # ignore test examples with ticker not in training set

  # combine the different datatype pipes
  preprocessor = ColumnTransformer(transformers=[
      ('num', num_transformer, num_features),
      ('cat', cat_transformer, cat_features)])

  # make the master classifier pipe
  clf_svc_base = SVC(kernel='linear', gamma='scale', class_weight='balanced')
  clf_svc = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('model', clf_svc_base)])

  # collect the data and split for testing
  all_features = np.hstack((cat_features, num_features, pred_weight))
  X = data_clean[all_features]
  y = data_clean[pred_feature]

  X_train_weight, X_test_weight, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
  X_train = X_train_weight.drop(pred_weight, axis=1)
  X_test = X_test_weight.drop(pred_weight, axis=1)

  # cross validate to find regularization parameter C = 1/lambda
  reg_param_name = 'C'
  param_space = np.logspace(-4, 0, 50)
  sample_weight = X_train_weight[pred_weight]

  score_params={'sample_weight': sample_weight}
  scorer = 'f1'#make_scorer(score_f1_weighted, **score_params)


  search = cv_gs_wrap(clf_svc, X_train, y_train, reg_param_name, param_space, sample_weight=None,
                      tss_splits=5, scorer=scorer, verbose=True)
  display_cv_metrics(search, reg_param_name, scorer)
  display_is_metrics(search, X_train, y_train)
  display_oos_metrics(search, X_test, y_test)
  display_feature_imp(search, num_features)
