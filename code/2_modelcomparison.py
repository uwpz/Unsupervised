# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm

# Specific libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # , GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression  # , ElasticNet
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

#  from sklearn.tree import DecisionTreeRegressor, plot_tree , export_graphviz

# Main parameter
TARGET_TYPE = "CLASS"

# Specific parameters
n_jobs = 4
metric = "auc"  # metric for peformance comparison

# Silent plotting (Overwrite to get default: plt.ion();  matplotlib.use('TkAgg'))
plt.ioff(); matplotlib.use('Agg')
plt.ion(); matplotlib.use('TkAgg')


# Load results from exploration
df = metr = cate = target_labels = None
with open(TARGET_TYPE + "_1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")

# Scale "metr_enocded" features for DL (Tree-based are not influenced by this Trafo)
df[metr] = (df[metr] - df[metr].min()) / (df[metr].max() - df[metr].min())


# ######################################################################################################################
# # Test an algorithm (and determine parameter grid)
# ######################################################################################################################

# --- Sample data ----------------------------------------------------------------------------------------------------

# Undersample only training data (take all but n_maxpersample at most)
under_samp = Undersample(n_max_per_level = 10000)
df_tmp = under_samp.fit_transform(df.query("fold == 'train'").reset_index())
b_all = under_samp.b_all
b_sample = under_samp.b_sample
print(b_sample, b_all)
df_tune = pd.concat([df_tmp, df.query("fold == 'test'").reset_index(drop = True)], sort = False).reset_index(
    drop = True)
df_tune.groupby("fold")["target"].describe()



# --- Define some splits -------------------------------------------------------------------------------------------

# split_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
split_my1fold_cv = TrainTestSep(1)
split_5fold = KFold(5, shuffle=True, random_state=42)
split_my5fold_cv = TrainTestSep(5)
split_my5fold_boot = TrainTestSep(5, "bootstrap")
'''
df_tune["fold"].value_counts()
mysplit = split_my5fold_cv.split(df_tune)
i_train, i_test = next(mysplit)
df_tune["fold"].iloc[i_train].describe()
df_tune["fold"].iloc[i_test].describe()
i_test.sort()
i_test
'''


# --- Fits -----------------------------------------------------------------------------------------------------------

# Lasso / Elastic Net
fit = (GridSearchCV(SGDRegressor(penalty = "ElasticNet", warm_start = True) if TARGET_TYPE == "REGR" else
                    SGDClassifier(loss = "log", penalty = "ElasticNet", warm_start = True),  # , tol=1e-2
                    {"alpha": [2 ** x for x in range(-4, -12, -1)],
                     "l1_ratio": [1]},
                    cv = split_my1fold_cv.split(df_tune),
                    refit = False,
                    scoring = d_scoring[TARGET_TYPE],
                    return_train_score = True,
                    n_jobs = n_jobs)
       .fit(CreateSparseMatrix(metr = metr, cate = cate, df_ref = df_tune).fit_transform(df_tune),
            df_tune["target"]))
plot_cvresult(fit.cv_results_, metric = metric, x_var = "alpha", color_var = "l1_ratio")
pd.DataFrame(fit.cv_results_)


# XGBoost
start = time.time()
fit = (GridSearchCV_xlgb(xgb.XGBRegressor(verbosity = 0) if TARGET_TYPE == "REGR" else xgb.XGBClassifier(verbosity = 0),
                         {"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01],
                          "max_depth": [3], "min_child_weight": [5]},
                         cv = split_my1fold_cv.split(df_tune),
                         refit = False,
                         scoring = d_scoring[TARGET_TYPE],
                         return_train_score = True,
                         n_jobs = n_jobs)
       .fit(CreateSparseMatrix(metr = metr, cate = cate, df_ref = df_tune).fit_transform(df_tune),
            df_tune["target"]))
print(time.time()-start)
pd.DataFrame(fit.cv_results_)
plot_cvresult(fit.cv_results_, metric = metric,
              x_var = "n_estimators", color_var = "max_depth", column_var = "min_child_weight")



# ######################################################################################################################
# Simulation: compare algorithms
# ######################################################################################################################

# Basic data sampling
df_modelcomp = df_tune.copy()


# --- Run methods ------------------------------------------------------------------------------------------------------

df_modelcomp_result = pd.DataFrame()  # intialize

# Elastic Net
cvresults = cross_validate(
    estimator = GridSearchCV(SGDRegressor(penalty = "ElasticNet", warm_start = True) if TARGET_TYPE == "REGR" else
                             SGDClassifier(loss = "log", penalty = "ElasticNet", warm_start = True),  # , tol=1e-2
                             {"alpha": [2 ** x for x in range(-4, -12, -1)],
                              "l1_ratio": [1]},
                             cv = ShuffleSplit(1, 0.2, random_state = 999),  # just 1-fold for tuning
                             refit = metric,
                             scoring = d_scoring[TARGET_TYPE],
                             return_train_score = False,
                             n_jobs = n_jobs),
    X = CreateSparseMatrix(metr = metr, cate = cate, df_ref = df_modelcomp).fit_transform(df_modelcomp),
    y = df_modelcomp["target"],
    cv = split_my5fold_cv.split(df_modelcomp),
    scoring = d_scoring[TARGET_TYPE],
    return_train_score = False,
    n_jobs = n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model = "ElasticNet"),
                                                 ignore_index = True)

# Xgboost
cvresults = cross_validate(
    estimator = GridSearchCV_xlgb(
        xgb.XGBRegressor(verbosity = 0) if TARGET_TYPE == "REGR" else xgb.XGBClassifier(verbosity = 0),
        {"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01],
         "max_depth": [3], "min_child_weight": [10]},
        cv = ShuffleSplit(1, 0.2, random_state = 999),  # just 1-fold for tuning
        refit = metric,
        scoring = d_scoring[TARGET_TYPE],
        return_train_score = False,
        n_jobs = n_jobs),
    X = CreateSparseMatrix(metr = metr, cate = cate, df_ref = df_modelcomp).fit_transform(
        df_modelcomp),
    y = df_modelcomp["target"],
    cv = split_my5fold_cv.split(df_modelcomp),
    scoring = d_scoring[TARGET_TYPE],
    return_train_score = False,
    n_jobs = n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model = "XGBoost"),
                                                 ignore_index = True)


# --- Plot model comparison ------------------------------------------------------------------------------

plot_modelcomp(df_modelcomp_result.rename(columns = {"index": "run", "test_" + metric: metric}),
               scorevar = metric,
               pdf = plotloc + TARGET_TYPE + "_model_comparison.pdf")


#plt.close("all")
