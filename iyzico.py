import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
import re

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
warnings.filterwarnings("ignore")

###############################################################
# Veri Setinin Keşfi
###############################################################

# 1. iyzico_data.csv dosyasının okutulması ve transaction_date değişkeninin tipini date'e çevrilmesi.
df = pd.read_csv("datasets/iyzico_data.csv", index_col=[0])
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# Kayıtların başlangıç ve bitiş tarihleri.
df["transaction_date"].min()  # Timestamp("2018-01-01 00:00:00")
df["transaction_date"].max()  # Timestamp("2020-12-31 00:00:00")

# Her üye iş yerindeki toplam işlem sayısı.
df.groupby("merchant_id").agg({"Total_Transaction": "sum"})

# Her üye iş yerindeki toplam ödeme miktarı.
df.groupby("merchant_id").agg({"Total_Paid": "sum"})

# Üye iş yerlerinin her bir yıl içerisindeki transaction count grafikleri.
for i in df.merchant_id.unique():
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 1, 1, title=str(i) + " 2018-2019 Transaction Count")
    df[(df.merchant_id == i) & (df.transaction_date >= "2018-01-01") & (df.transaction_date < "2019-01-01")][
        "Total_Transaction"].plot()
    plt.subplot(2, 1, 2, title=str(i) + " 2019-2020 Transaction Count")
    df[(df.merchant_id == i) & (df.transaction_date >= "2019-01-01") & (df.transaction_date < "2020-01-01")][
        "Total_Transaction"].plot()
    plt.show(block=True)


###############################################################
# Feature Engineering
###############################################################

########################
# Date Features
########################

def create_date_features(dataframe, date_column):
    dataframe["month"] = dataframe[date_column].dt.month
    dataframe["day_of_month"] = dataframe[date_column].dt.day
    dataframe["day_of_year"] = dataframe[date_column].dt.dayofyear
    dataframe["week_of_year"] = dataframe[date_column].dt.weekofyear
    dataframe["day_of_week"] = dataframe[date_column].dt.dayofweek
    dataframe["year"] = dataframe[date_column].dt.year
    dataframe["is_wknd"] = dataframe[date_column].dt.weekday // 4
    dataframe["is_month_start"] = dataframe[date_column].dt.is_month_start.astype(int)
    dataframe["is_month_end"] = dataframe[date_column].dt.is_month_end.astype(int)
    dataframe["quarter"] = dataframe[date_column].dt.quarter
    dataframe["is_quarter_start"] = dataframe[date_column].dt.is_quarter_start.astype(int)
    dataframe["is_quarter_end"] = dataframe[date_column].dt.is_quarter_end.astype(int)
    dataframe["is_year_start"] = dataframe[date_column].dt.is_year_start.astype(int)
    dataframe["is_year_end"] = dataframe[date_column].dt.is_year_end.astype(int)
    return dataframe


df = create_date_features(df, "transaction_date")

# Üye iş yerlerinin yıl ve ay bazında işlem sayılarının incelenmesi.
df.groupby(["merchant_id", "year", "month"]).agg({"Total_Transaction": ["sum", "mean", "median"]})

# Üye iş yerlerinin yıl ve ay bazında toplam ödeme miktarlarının incelenmesi.
df.groupby(["merchant_id", "year", "month"]).agg({"Total_Paid": ["sum", "mean", "median"]})


########################
# Lag/Shifted Features
########################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe["sales_lag_" + str(lag)] = dataframe.groupby(["merchant_id"])["Total_Transaction"].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91, 181, 271, 361, 391, 481, 571, 661, 730])


########################
# Rolling Mean Features
########################

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe["sales_roll_mean_" + str(window)] = dataframe.groupby("merchant_id")["Total_Transaction"]. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [91, 181, 271, 361, 391, 481, 571, 661, 730])


########################
# Exponentially Weighted Mean Features
########################

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe["sales_ewm_alpha_" + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby("merchant_id")["Total_Transaction"].transform(
                    lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas_ = [0.95, 0.85, 0.75, 0.65, 0.55]
lags_ = [91, 181, 271, 361, 391, 481, 571, 661, 730]

df = ewm_features(df, alphas_, lags_)

########################
# Black Friday - Summer Solstice - School Start
########################

df["is_black_friday"] = 0
df.loc[df["transaction_date"].isin(["2018-11-22", "2018-11-23", "2019-11-29", "2019-11-30"]), "is_black_friday"] = 1

df["is_summer_solstice"] = 0
df.loc[df["transaction_date"].isin(["2018-06-19", "2018-06-20", "2018-06-21", "2018-06-22",
                                    "2019-06-19", "2019-06-20", "2019-06-21",
                                    "2019-06-22", ]), "is_summer_solstice"] = 1

df["is_school_start"] = 0
df.loc[((df["transaction_date"] > "2018-09-12") & (df["transaction_date"] > "2018-09-21"))
       | ((df["transaction_date"] > "2018-09-02") & (df["transaction_date"] > "2018-09-16")), "is_school_start"] = 1

########################
# One-Hot Encoding
########################
df.head()

df = pd.get_dummies(df, columns=["merchant_id", "day_of_week", "month"])
df["Total_Transaction"] = np.log1p(df["Total_Transaction"].values)


########################
# Custom Cost Function
########################

# SMAPE: Symmetric mean absolute percentage error.

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return "SMAPE", smape_val, False


########################
# Time-Based Validation Sets
########################

df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

# 2019'un 10.ayına kadar train seti.
train = df.loc[(df["transaction_date"] < "2019-10-01"), :]

# 2019'un son 3 ayı validasyon seti.
val = df.loc[(df["transaction_date"] >= "2019-10-01") & (df["transaction_date"] < "2020-01-01"), :]

cols = [col for col in train.columns if col not in ["transaction_date", "Total_Transaction", "Total_Paid", "year"]]

Y_train = train["Total_Transaction"]
X_train = train[cols]

Y_val = val["Total_Transaction"]
X_val = val[cols]

# Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################
# LightGBM Model
########################

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

num_leaves = [3, 4, 5]
learning_rate = [0.15, 0.2, 0.25]
feature_fraction = [0.7, 0.75, 0.8]
max_depth = [3, 4, 5]


def hyperparameter_opt(trainset, valset, num_leaves_, learning_rate_, feature_fraction_, max_depth_):
    record = float("inf")
    parameter = {}
    for num in num_leaves_:
        for rate in learning_rate_:
            for fraction in feature_fraction_:
                for depth in max_depth_:
                    print("num_leaves " + str(num) + " learning_rate " + str(rate) +
                          " feature_fraction " + str(fraction) + " max_depth " + str(depth))
                    parameters = {"metric": {"mae"}, "num_leaves": num, "learning_rate": rate,
                                  "feature_fraction": fraction, "max_depth": depth, "verbose": -1,
                                  "num_boost_round": 10000, "nthread": -1}
                    model_ = lgb.train(parameters, trainset,
                                       valid_sets=[trainset, valset],
                                       num_boost_round=parameters["num_boost_round"],
                                       feval=lgbm_smape,
                                       callbacks=[lgb.early_stopping(200, verbose=False)])
                    pred_values = model_.predict(X_val, num_iteration=model_.best_iteration)
                    error = smape(np.expm1(pred_values), np.expm1(Y_val))
                    print(error)
                    if error < record:
                        record = error
                        parameter = parameters
    return parameter, record


best_parameters, error = hyperparameter_opt(lgbtrain, lgbval, num_leaves, learning_rate, feature_fraction, max_depth)

model = lgb.train(best_parameters, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=10000,
                  feval=lgbm_smape,
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)])

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))


########################
# Değişken önem düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance("gain")
    feat_imp = pd.DataFrame({"feature": model.feature_name(),
                             "split": model.feature_importance("split"),
                             "gain": 100 * gain / gain.sum()}).sort_values("gain", ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title("feature")
        plt.tight_layout()
        plt.show(block=True)
    else:
        return feat_imp


plot_lgb_importances(model, num=30, plot=True)

# lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
# plt.show(block=True)

feat_imp = plot_lgb_importances(model, num=200)
importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values


########################
# Final Model
########################
# 2020'nin 10.ayına kadar train seti.
train = df.loc[df["transaction_date"] < "2020-10-01"]
Y_train = train["Total_Transaction"]
X_train = train[cols]

# 2020'nin son 3 ayı tahmin seti.
test = df.loc[df["transaction_date"] > "2020-10-01"]
X_test = test[cols]
real_values = test["Total_Transaction"]

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(best_parameters, lgbtrain_all, num_boost_round=model.best_iteration)

pred_val = final_model.predict(X_test, num_iteration=model.best_iteration)

smape(np.expm1(pred_val), np.expm1(real_values))  # 27.298551211816704
