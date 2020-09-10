# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *

# sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm

# Specific libraries


# Main parameter
TARGET_TYPE = "CLASS"

# Specific parameters (CLASS is default)
ylim = None
cutoff_corr = 0.1
cutoff_varimp = 0.52
color = twocol
min_width = 0



# ######################################################################################################################
# ETL
# ######################################################################################################################

# --- Read data ------------------------------------------------------------------------------------------------------

df_orig = pd.read_csv(dataloc + "credit_card.csv")
df_orig.describe()
df_orig.columns

# "Save" original data
df = df_orig.copy()


# --- Read metadata (Project specific) -----------------------------------------------------------------------------



# --- Feature engineering -----------------------------------------------------------------------------------------

# TBD



# --- Define target and train/test-fold ----------------------------------------------------------------------------

# Target
df["target"] = df["Class"]
df["target"].value_counts()

# Train/Test fold: usually split by time
np.random.seed(123)
df["fold"] = np.random.permutation(
    pd.qcut(np.arange(len(df)), q = [0, 0.8, 1], labels = ["train", "test"]))
print(df.fold.value_counts())
df["fold_num"] = df["fold"].replace({"train": 0, "test": 1})  # Used for pedicting test data

# Define the id
df["id"] = np.arange(len(df)) + 1


# ######################################################################################################################
# Metric variables: Explore and adapt
# ######################################################################################################################

# --- Define metric covariates -------------------------------------------------------------------------------------

metr = np.array(["Time", "Amount"] + ["V" + str(x) for x in (np.arange(28) + 1)]).astype(object)
df[metr].describe()


# --- Missings + Outliers + Skewness ---------------------------------------------------------------------------------

# Remove covariates with too many missings from metr
misspct = df[metr].isnull().mean().round(3)  # missing percentage
print("misspct:\n", misspct.sort_values(ascending = False))  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
metr = setdiff(metr, remove)  # adapt metadata

# Check for outliers and skewness
df[metr].describe()
plot_distr(df, metr, target_type = TARGET_TYPE,
           color = color, ylim = ylim,
           ncol = 4, nrow = 2, w = 18, h = 12, pdf = plotloc + TARGET_TYPE + "_distr_metr.pdf")

'''
# Winsorize (hint: plot again before deciding for log-trafo)
df = Winsorize(features = metr, lower_quantile = 0.001, upper_quantile = 0.999).fit_transform(df)

# Log-Transform
if TARGET_TYPE == "CLASS":
    tolog = np.array(["fare"], dtype = "object")
else:
    tolog = np.array(["Lot_Area"], dtype = "object")
df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - min(0, np.min(x)) + 1))
metr = np.where(np.isin(metr, tolog), metr + "_LOG_", metr)  # adapt metadata (keep order)
df.rename(columns = dict(zip(tolog + "_BINNED", tolog + "_LOG_" + "_BINNED")), inplace = True)  # adapt binned version
'''

# --- Final variable information ------------------------------------------------------------------------------------

# Univariate variable importance
varimp_metr = calc_imp(df, metr, target_type = TARGET_TYPE)
print(varimp_metr)

# Plot
plot_distr(df, features = varimp_metr.index.values, target_type = TARGET_TYPE,
           varimp = varimp_metr, color = color, ylim = ylim,
           ncol = 4, nrow = 2, w = 24, h = 18, pdf = plotloc + TARGET_TYPE + "_distr_metr_final.pdf")


# --- Correlation -------------------------------------------------------------------------------------------

plot_corr(df, metr, cutoff = 0, pdf = plotloc + TARGET_TYPE + "_corr_metr.pdf", w = 18, h = 18)


# --- Time/fold depedency --------------------------------------------------------------------------------------------

# Univariate variable importance (again ONLY for non-missing observations!)
varimp_metr_fold = calc_imp(df, metr, "fold_num")


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Adapt target ----------------------------------------------------------------------------------------

# Switch target to numeric in case of multiclass
target_labels = "target"


# --- Save image ----------------------------------------------------------------------------------------------------

# Clean up
plt.close(fig = "all")  # plt.close(plt.gcf())
del df_orig

# Serialize
with open(TARGET_TYPE + "_1_explore.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "target_labels": target_labels,
                 "metr": metr,
                 "cate": None},
                file)
