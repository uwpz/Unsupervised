# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm

# Specific libraries
from sklearn.model_selection import cross_validate

# Main parameter
TARGET_TYPE = "CLASS"

# Specific parameters
n_jobs = 4
labels = None
metric = "auc"  # metric for peformance comparison
importance_cut = 99
topn = 8
ylim_res = (0, 1)
color = twocol

# Silent plotting (Overwrite to get default: plt.ion();  matplotlib.use('TkAgg'))
plt.ioff(); matplotlib.use('Agg')
plt.ion(); matplotlib.use('TkAgg')

# Load results from exploration
df = metr = cate = target_labels = None
with open(TARGET_TYPE + "_1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")

# Features for xgboost
features = metr

'''
np.random.seed(124)
df["fold"] = np.random.permutation(
    pd.qcut(np.arange(len(df)), q = [0, 0.8, 1], labels = ["train", "test"]))
print(df.fold.value_counts())
df["fold_num"] = df["fold"].replace({"train": 0, "test": 1})  # Used for pedicting test data
'''

# ######################################################################################################################
# Prepare
# ######################################################################################################################

# Tuning parameter to use (for xgb) and classifier definition
xgb_param = dict(n_estimators = 700, learning_rate = 0.01,
                 max_depth = 3, min_child_weight = 5,
                 colsample_bytree = 1, subsample = 1,
                 gamma = 0,
                 verbosity = 0,
                 n_jobs = n_jobs)
clf = xgb.XGBRegressor(**xgb_param) if TARGET_TYPE == "REGR" else xgb.XGBClassifier(**xgb_param)


# --- Sample data ----------------------------------------------------------------------------------------------------

if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    # Training data: Just take data from train fold (take all but n_maxpersample at most)
    df.loc[df["fold"] == "train", "target"].describe()
    under_samp = Undersample(n_max_per_level = 10000)
    df_train = under_samp.fit_transform(df.query("fold == 'train'").reset_index(drop = True))
    b_sample = under_samp.b_sample
    b_all = under_samp.b_all
    print(b_sample, b_all)
else:
    df_train = (df.query("fold == 'train'").sample(n = min(df.query("fold == 'train'").shape[0], int(5e3)))
                .reset_index(drop = True))
    b_sample = None
    b_all = None

# Test data
df_test = df.query("fold == 'test'").reset_index(drop = True)  # .sample(300) #ATTENTION: Do not sample in final run!!!

# Combine again
df_traintest = pd.concat([df_train, df_test]).reset_index(drop = True)

# Folds for crossvalidation and check
split_my5fold = TrainTestSep(5, "cv")
for i_train, i_test in split_my5fold.split(df_traintest):
    print("TRAIN-fold:", df_traintest["fold"].iloc[i_train].value_counts())
    print("TEST-fold:", df_traintest["fold"].iloc[i_test].value_counts())
    print("##########")


# ######################################################################################################################
# Performance
# ######################################################################################################################

# --- Do the full fit and predict on test data -------------------------------------------------------------------

# Fit
tr_spm = CreateSparseMatrix(metr = metr, cate = cate, df_ref = df_traintest)
X_train = tr_spm.fit_transform(df_train)
fit = clf.fit(X_train, df_train["target"].values)

# Predict
X_test = tr_spm.transform(df_test)
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    yhat_test = scale_predictions(fit.predict_proba(X_test), b_sample, b_all)
else:
    yhat_test = fit.predict(X_test)
print(pd.DataFrame(yhat_test).describe())

# Performance
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    print(auc(df_test["target"].values, yhat_test))
else:
    print(spear(df_test["target"].values, yhat_test))

# Plot performance
plot_all_performances(df_test["target"], yhat_test, target_labels = target_labels, target_type = TARGET_TYPE,
                      color = color, ylim = None,
                      pdf = plotloc + TARGET_TYPE + "_performance.pdf")


# --- Check performance for crossvalidated fits ---------------------------------------------------------------------
d_cv = cross_validate(clf, tr_spm.transform(df_traintest), df_traintest["target"],
                      cv = split_my5fold.split(df_traintest),  # special 5fold
                      scoring = d_scoring[TARGET_TYPE],
                      return_estimator = True,
                      n_jobs = 4)
# Performance
print(d_cv["test_" + metric])
print(d_cv["test_" + "prec_rec_auc"])


# ######################################################################################################################
# Variable Importance
# ######################################################################################################################

# --- Variable Importance by permuation argument -------------------------------------------------------------------
# Importance for "total" fit (on test data!)
df_varimp = calc_varimp_by_permutation(df_test, fit, tr_spm = tr_spm, features = metr,
                                       target_type = TARGET_TYPE,
                                       b_sample = b_sample, b_all = b_all, n_jobs = 1)
topn_features = df_varimp["feature"].values[range(topn)]

# Add other information (e.g. special category): category variable is needed -> fill with at least with "dummy"
df_varimp["Category"] = pd.cut(df_varimp["importance"], [-np.inf, 10, 50, np.inf], labels = ["low", "medium", "high"])

# Plot
plot_variable_importance(df_varimp, mask = df_varimp["feature"].isin(topn_features),
                         pdf = plotloc + TARGET_TYPE + "_variable_importance.pdf")

#plt.close("all")



