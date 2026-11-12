# =============================================================================
#  ML LAB EXAM - COPY/PASTE TEMPLATE
#  Each "# %%" marks one notebook cell. Copy what you need, edit names.
#
#  Required answer points (every question):
#   1. Type of method + justification
#   2. Exploratory analysis
#   3. Preprocessing / feature selection
#   4. ML pipeline
#   5. Results
#   6. Ethical concerns
#
#  PART 0-3  = COMMON (same for every problem)
#  PART 4    = METHODS (pick ONE: A classification, B regression, C clustering,
#              D NLP / text classification, E web scraping, F time series,
#              G PCA / dimensionality reduction)
#  PART 5    = Ethics bank + write-up templates
#
#  EDIT: 'data.csv', 'target', column names in the lists.
# =============================================================================


# %% ============================ PART 0: IMPORTS ============================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, KFold, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve,
                             mean_absolute_error, mean_squared_error, r2_score,
                             silhouette_score)

# Clustering / dimensionality reduction
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

RANDOM_STATE = 42


# %% ========================= PART 1: LOAD THE DATASET ======================
df = pd.read_csv("data.csv")              # CSV
# df = pd.read_excel("data.xlsx")         # Excel
# df = pd.read_csv("data.csv", sep=";")   # different separator
# df = pd.read_json("data.json")          # JSON
# from sklearn.datasets import load_iris  # built-in sklearn dataset
# d = load_iris(as_frame=True); df = d.frame

TARGET = "target"        # <-- EDIT: name of the label column (None if unsupervised)
df.head()


# %% ====================== PART 2: EXPLORATORY DATA ANALYSIS ================
# (Write: size, dtypes, missing values, duplicates, imbalance, outliers, correlations)
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\n--- info ---");      df.info()
print("\n--- describe ---");  print(df.describe(include="all").T)
print("\n--- missing values ---");  print(df.isnull().sum())
print("\n--- % missing ---");       print((df.isnull().mean() * 100).round(2))
print("\nDuplicates:", df.duplicated().sum())

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
print("\nNumeric:", num_cols)
print("Categorical:", cat_cols)

# Class balance (classification only)
if TARGET in df.columns:
    print("\n--- target distribution ---")
    print(df[TARGET].value_counts())
    print(df[TARGET].value_counts(normalize=True).round(3))
    sns.countplot(x=TARGET, data=df); plt.title("Target distribution"); plt.show()


# %% EDA plots
df[num_cols].hist(figsize=(12, 8), bins=20); plt.tight_layout(); plt.show()      # distributions

plt.figure(figsize=(12, 5)); sns.boxplot(data=df[num_cols]); plt.xticks(rotation=45)
plt.title("Outliers (boxplot)"); plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap"); plt.show()

# sns.pairplot(df, hue=TARGET)                         # features vs target (small data only)
# sns.boxplot(x=TARGET, y="feature", data=df)          # feature per class
# df.corr(numeric_only=True)[TARGET].sort_values()     # correlation with numeric target


# %% ======================= PART 3: COMMON PREPROCESSING ====================
# (Write what you did AND why)

# 3.1 Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# 3.2 Drop useless columns (IDs, names, constant columns)
# df = df.drop(columns=["id", "name"])

# 3.3 Missing values
for c in df.select_dtypes(include=np.number).columns:
    df[c] = df[c].fillna(df[c].median())            # numeric -> median (robust to outliers)
for c in df.select_dtypes(exclude=np.number).columns:
    df[c] = df[c].fillna(df[c].mode()[0])           # categorical -> mode
# df = df.dropna()                                  # alternative: drop rows if very few missing

# 3.4 Outliers (IQR rule) - optional
def remove_outliers_iqr(data, cols):
    for c in cols:
        q1, q3 = data[c].quantile([0.25, 0.75]); iqr = q3 - q1
        data = data[(data[c] >= q1 - 1.5 * iqr) & (data[c] <= q3 + 1.5 * iqr)]
    return data
# df = remove_outliers_iqr(df, ["col1", "col2"])

# 3.5 Encode categorical features
cat_features = [c for c in df.select_dtypes(exclude=np.number).columns if c != TARGET]
df = pd.get_dummies(df, columns=cat_features, drop_first=True)     # one-hot

# Encode a categorical TARGET (classification)
# le = LabelEncoder(); df[TARGET] = le.fit_transform(df[TARGET]); print(le.classes_)

# 3.6 Split X / y
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Classification: add stratify=y.  Regression: remove stratify.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# 3.7 Scaling (needed: KNN, SVM, LogReg, KMeans, PCA, Ridge/Lasso. NOT needed: trees/forests)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)      # fit on TRAIN only (avoids data leakage)
X_test_s = scaler.transform(X_test)

# 3.8 Feature selection (optional)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
# sel = SelectKBest(f_classif, k=5).fit(X_train, y_train)      # f_regression for regression
# print(X.columns[sel.get_support()])
# Drop one of each highly correlated pair (>0.9):
# corr = X.corr().abs(); upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
# drop = [c for c in upper.columns if any(upper[c] > 0.9)]; X = X.drop(columns=drop)
# Tree importance (after fitting a forest): pd.Series(model.feature_importances_, X.columns).sort_values().plot.barh()

# 3.9 Class imbalance (if target very uneven)
# Option 1: model = RandomForestClassifier(class_weight="balanced")
# Option 2: from imblearn.over_sampling import SMOTE; X_train, y_train = SMOTE().fit_resample(X_train, y_train)


# =============================================================================
#  PART 4: METHODS  (pick the one that matches the question)
# =============================================================================

# ----------------------------------------------------------------------------
# A. CLASSIFICATION  (target is a category: spam/not spam, disease yes/no, species)
# ----------------------------------------------------------------------------
# 1. METHOD: "Supervised classification - labelled, categorical target `TARGET`.
#    Models compared: Logistic Regression (baseline, interpretable), Decision Tree
#    (interpretable), Random Forest (non-linear, robust), KNN/SVM (scaled)."
# 2-3. EDA + preprocessing: Part 2 and Part 3 above.

# %% A4. Pipeline: train & compare several models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True),
    "NaiveBayes": GaussianNB(),
}
scaled_models = {"LogisticRegression", "KNN", "SVM"}      # these use scaled data
results = {}
for name, m in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if name in scaled_models else (X_train, X_test)
    m.fit(Xtr, y_train)
    pred = m.predict(Xte)
    results[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, pred, average="weighted"),
        "F1": f1_score(y_test, pred, average="weighted"),
    }
res_df = pd.DataFrame(results).T.sort_values("F1", ascending=False)
print(res_df)

# %% A5. Results for the best model
best_name = res_df.index[0]
best = models[best_name]
Xte = X_test_s if best_name in scaled_models else X_test
pred = best.predict(Xte)
print("Best model:", best_name)
print(classification_report(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues"); plt.show()

# Cross-validation (more reliable than a single split)
Xcv = X_train_s if best_name in scaled_models else X_train
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
print("CV accuracy:", cross_val_score(best, Xcv, y_train, cv=cv).mean())

# Hyperparameter tuning (example for Random Forest)
# grid = GridSearchCV(RandomForestClassifier(random_state=42),
#                     {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}, cv=5, scoring="f1_weighted")
# grid.fit(X_train, y_train); print(grid.best_params_, grid.best_score_)

# ROC-AUC (binary only)
# print(roc_auc_score(y_test, best.predict_proba(Xte)[:, 1]))

# Feature importance (tree models)
# pd.Series(models["RandomForest"].feature_importances_, index=X.columns).sort_values().plot.barh(); plt.show()

# Interpretation to write: accuracy vs F1 (if imbalanced, accuracy misleads), which class is
# confused (confusion matrix), false negatives vs false positives cost, how to improve.


# ----------------------------------------------------------------------------
# B. REGRESSION  (target is a number: price, salary, temperature)
# ----------------------------------------------------------------------------
# 1. METHOD: "Supervised regression - continuous numeric target. Linear Regression as a
#    interpretable baseline; Random Forest for non-linear relationships;
#    Ridge/Lasso to reduce overfitting/for feature selection."
# NOTE: for regression use train_test_split WITHOUT stratify.

# %% B4. Pipeline
reg_models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "KNN": KNeighborsRegressor(n_neighbors=5),
}
reg_scaled = {"Ridge", "Lasso", "KNN"}
reg_results = {}
for name, m in reg_models.items():
    Xtr, Xte = (X_train_s, X_test_s) if name in reg_scaled else (X_train, X_test)
    m.fit(Xtr, y_train)
    p = m.predict(Xte)
    reg_results[name] = {"MAE": mean_absolute_error(y_test, p),
                         "RMSE": np.sqrt(mean_squared_error(y_test, p)),
                         "R2": r2_score(y_test, p)}
reg_df = pd.DataFrame(reg_results).T.sort_values("R2", ascending=False)
print(reg_df)

# %% B5. Results / diagnostics
best_name = reg_df.index[0]; best = reg_models[best_name]
Xte = X_test_s if best_name in reg_scaled else X_test
p = best.predict(Xte)

plt.scatter(y_test, p, alpha=0.6); plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Actual vs Predicted"); plt.show()

resid = y_test - p
plt.scatter(p, resid, alpha=0.6); plt.axhline(0, color="r"); plt.xlabel("Predicted"); plt.ylabel("Residual")
plt.title("Residual plot"); plt.show()

# Linear model coefficients (interpretation)
# print(pd.Series(reg_models["Linear"].coef_, index=X.columns).sort_values())
# print("CV R2:", cross_val_score(best, X_train, y_train, cv=5, scoring="r2").mean())

# Interpretation: R2 = share of variance explained; RMSE in target units; residuals random =>
# good fit; pattern in residuals => non-linear relation / missing features.


# ----------------------------------------------------------------------------
# C. CLUSTERING  (NO target column: customer segmentation, grouping)
# ----------------------------------------------------------------------------
# 1. METHOD: "Unsupervised learning - no labels, goal is to discover groups. K-Means chosen
#    (simple, scalable); k chosen by elbow + silhouette. Hierarchical/DBSCAN as alternatives."
# Preprocessing: NO train/test split needed. Scaling is MANDATORY. Use numeric features only.

# %% C3. Prepare features
features = ["col1", "col2"]                       # <-- EDIT (or: df.select_dtypes('number').columns)
Xc = df[features]
Xc_s = StandardScaler().fit_transform(Xc)

# %% C4. Pipeline: choose k (elbow + silhouette)
inertia, sil = [], []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xc_s)
    inertia.append(km.inertia_); sil.append(silhouette_score(Xc_s, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(K, inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
ax[1].plot(K, sil, "o-"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k"); plt.show()

# %% C5. Final model + results
k = 3                                             # <-- EDIT from elbow/silhouette
km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xc_s)
df["cluster"] = km.labels_
print("Silhouette:", silhouette_score(Xc_s, km.labels_))
print(df["cluster"].value_counts())
print(df.groupby("cluster")[features].mean())     # CLUSTER PROFILES -> name each segment

# Visualise in 2D with PCA
p2 = PCA(2).fit_transform(Xc_s)
plt.scatter(p2[:, 0], p2[:, 1], c=km.labels_, cmap="viridis", alpha=0.7)
plt.title("Clusters (PCA 2D)"); plt.show()

# Alternatives:
# AgglomerativeClustering(n_clusters=3).fit_predict(Xc_s)
# DBSCAN(eps=0.5, min_samples=5).fit_predict(Xc_s)     # label -1 = noise/outlier
# from scipy.cluster.hierarchy import dendrogram, linkage
# dendrogram(linkage(Xc_s, "ward")); plt.show()
# Interpretation: describe each cluster in business terms (e.g. "high income, low spend").


# ----------------------------------------------------------------------------
# D. NLP / TEXT  (tokenise, POS tag, WordNet, text classification)
# ----------------------------------------------------------------------------
# 1. METHOD: "Natural Language Processing. Text is converted to numbers with TF-IDF, then a
#    supervised classifier (Naive Bayes / Logistic Regression) is trained."

# %% D0. NLTK setup
import re, nltk
nltk.download("punkt"); nltk.download("punkt_tab"); nltk.download("stopwords")
nltk.download("wordnet"); nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger"); nltk.download("averaged_perceptron_tagger_eng")
nltk.download("brown")
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet, brown
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import FreqDist

# %% D3. Text preprocessing (the "preprocessing" step for text)
stop = set(stopwords.words("english"))
lemm = WordNetLemmatizer()
stem = PorterStemmer()

def clean_text(text):
    text = text.lower()                                   # lowercase
    text = re.sub(r"http\S+|[^a-z\s]", " ", text)         # remove urls, digits, punctuation
    tokens = word_tokenize(text)                          # tokenise
    tokens = [t for t in tokens if t not in stop and len(t) > 2]   # stopwords
    tokens = [lemm.lemmatize(t) for t in tokens]          # lemmatise (or stem.stem(t))
    return " ".join(tokens)

# df["clean"] = df["text"].apply(clean_text)              # <-- EDIT column name

# %% D-explore. Text EDA & NLTK basics
sample = "The cat is black"
tokens = word_tokenize(sample.lower())
print(nltk.pos_tag(tokens))                               # POS tags
# fd = FreqDist(" ".join(df["clean"]).split()); print(fd.most_common(10)); fd.plot(20)
# df["text"].str.len().hist()                             # length distribution
# from wordcloud import WordCloud
# plt.imshow(WordCloud().generate(" ".join(df["clean"]))); plt.axis("off")
# WordNet:
for s in wordnet.synsets("bank")[:3]:
    print(s.name(), s.pos(), s.definition(), s.examples())
# synonyms = {l.name() for s in wordnet.synsets("good") for l in s.lemmas()}
# antonyms = {a.name() for s in wordnet.synsets("good") for l in s.lemmas() for a in l.antonyms()}
# Brown corpus: brown.sents(categories="news"), brown.tagged_words(); nouns = tag.startswith("NN")

# %% D4. Pipeline: text classification
# Xt_train, Xt_test, yt_train, yt_test = train_test_split(df["clean"], df[TARGET],
#                                          test_size=0.2, random_state=42, stratify=df[TARGET])
# text_clf = Pipeline([
#     ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
#     ("clf", MultinomialNB()),                           # or LogisticRegression(max_iter=1000)
# ])
# text_clf.fit(Xt_train, yt_train)
# pred = text_clf.predict(Xt_test)
# print(classification_report(yt_test, pred))
# ConfusionMatrixDisplay.from_predictions(yt_test, pred); plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Sentiment without training: from nltk.sentiment import SentimentIntensityAnalyzer
#   nltk.download("vader_lexicon"); SentimentIntensityAnalyzer().polarity_scores(text)
# Interpretation: Bag-of-words ignores word order/context; TF-IDF down-weights common words.


# ----------------------------------------------------------------------------
# E. WEB SCRAPING  (data comes from a website -> then EDA / any model above)
# ----------------------------------------------------------------------------
# 1. METHOD: "Data acquisition by web scraping with BeautifulSoup, then cleaning (and optional
#    analysis)."

# %% E4. Pipeline
from bs4 import BeautifulSoup
import urllib.request as urllib2

url = "https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population"   # <-- EDIT
req = urllib2.Request(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(urllib2.urlopen(req), "lxml")          # or "html.parser"

def extract_txt(cell):
    parts = [c.strip() for c in cell.find_all(string=True) if "[" not in c]   # drop [1] citations
    return " ".join(parts).strip()

tables = soup.find_all("table", {"class": "wikitable"})
print(len(tables), "tables found")
table = tables[0]                                           # <-- choose correct table index

cols = [extract_txt(th) for th in table.find_all("tr")[0].find_all("th")]
rows = []
for tr in table.find_all("tr")[1:]:
    cells = tr.find_all(["th", "td"])
    if cells:
        rows.append([extract_txt(c) for c in cells])
df_web = pd.DataFrame(rows)
# df_web.columns = cols                                     # only if lengths match
print(df_web.head())

# Cleaning scraped data
# df_web["pop"] = df_web["pop"].str.replace(",", "").astype(int)
# df_web["pct"] = df_web["pct"].str.replace("%", "").astype(float)
# Alternatives: pd.read_html(url)[0]  (fastest for tables)
# soup.find_all("a"), soup.find("div", class_="x"), soup.select("css selector"), tag.get("href")
# import requests; r = requests.get(url, headers=...); BeautifulSoup(r.text, "html.parser")
# Then: continue with PART 2 EDA on df_web, then any method above.
# Ethics: robots.txt, terms of service, rate limits (time.sleep), copyright, personal data.


# ----------------------------------------------------------------------------
# F. TIME SERIES  (data ordered in time: sales, stock, temperature)
# ----------------------------------------------------------------------------
# 1. METHOD: "Time-series forecasting. Order matters, so split chronologically (no shuffle).
#    ARIMA / Holt-Winters or regression on lag features."

# %% F3. Preprocessing
ts = pd.read_csv("data.csv", parse_dates=["date"], index_col="date")   # <-- EDIT
ts = ts.sort_index()
series = ts["value"]                                                    # <-- EDIT
series = series.asfreq("D").interpolate()      # fix frequency ('D','W','M') & fill gaps

series.plot(figsize=(12, 4), title="Series"); plt.show()
series.rolling(30).mean().plot(label="30-pt rolling mean"); plt.legend(); plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(series, period=12).plot(); plt.show()   # trend / seasonality / residual
from statsmodels.tsa.stattools import adfuller
print("ADF p-value:", adfuller(series.dropna())[1])          # p<0.05 => stationary
# Non-stationary -> difference: series.diff().dropna()

# %% F4. Pipeline: chronological split + model
split = int(len(series) * 0.8)
train, test = series[:split], series[split:]                  # NO shuffling

from statsmodels.tsa.arima.model import ARIMA
fit = ARIMA(train, order=(1, 1, 1)).fit()                     # (p, d, q)
fc = fit.forecast(steps=len(test))

# Alternative: Holt-Winters
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# fc = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(len(test))
# Alternative: lag features -> any regression model
# lag_df = pd.DataFrame({"y": series, "lag1": series.shift(1), "lag7": series.shift(7)}).dropna()

# %% F5. Results
print("MAE :", mean_absolute_error(test, fc))
print("RMSE:", np.sqrt(mean_squared_error(test, fc)))
plt.figure(figsize=(12, 4)); plt.plot(train, label="train"); plt.plot(test, label="test")
plt.plot(test.index, fc, label="forecast"); plt.legend(); plt.show()
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # choose p and q


# ----------------------------------------------------------------------------
# G. DIMENSIONALITY REDUCTION (PCA)  - many features, visualisation, feature reduction
# ----------------------------------------------------------------------------
# 1. METHOD: "Unsupervised dimensionality reduction (PCA) to reduce correlated features while
#    keeping most variance; can feed a classifier or clustering."
# %% G4
pca_full = PCA().fit(X_train_s)
cum = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cum) + 1), cum, "o-"); plt.axhline(0.95, color="r", ls="--")
plt.xlabel("Components"); plt.ylabel("Cumulative explained variance"); plt.show()

pca = PCA(n_components=0.95)                 # keep 95% of variance
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)
print("Components kept:", pca.n_components_)
# Then train any model from A/B on X_train_p, X_test_p and compare with the non-PCA result.


# =============================================================================
#  PART 5: ETHICS BANK + WRITE-UP TEMPLATES
# =============================================================================
# Pick 3-4 that fit the dataset and tie each to the specific data:
#
#  - BIAS / FAIRNESS: training data may under-represent groups (gender, race, age,
#    region) so predictions can discriminate. Check performance per subgroup.
#  - PRIVACY / CONSENT: personal or sensitive data (health, finance, location);
#    anonymise, minimise data, comply with GDPR, secure storage.
#  - DATA PROVENANCE / LEGALITY: scraping needs robots.txt + terms of service
#    compliance, copyright, rate limiting.
#  - TRANSPARENCY / EXPLAINABILITY: black-box models in high-stakes areas
#    (health, loans, hiring) need explanations, humans in the loop.
#  - COST OF ERRORS: false negatives (missed disease) vs false positives (false fraud
#    flag) harm people differently -> choose metrics (recall/precision) accordingly.
#  - CLASS IMBALANCE / REPRESENTATIVENESS: model may ignore minority class; data may
#    not reflect the real population -> poor generalisation.
#  - SENSITIVE / PROXY FEATURES: remove or audit features that act as proxies
#    (zip code -> race, name -> gender).
#  - TEXT/NLP: toxic or biased language in corpora, user-generated content privacy.
#  - CLUSTERING: labelling people into segments can stereotype or exclude them.
#  - ACCOUNTABILITY & MISUSE: who is responsible for model decisions; monitoring
#    for drift; data leakage and over-claiming accuracy.
#
# WRITE-UP TEMPLATES
#  Method : "The dataset has labelled target `X` that is [categorical/continuous] -> supervised
#            [classification/regression]. I chose [model] because [reason]."
#  EDA    : "Dataset has N rows, M columns; dtypes ...; missing values in ...; [no] duplicates;
#            target is [balanced/imbalanced]; outliers in ...; strongest correlation: ..."
#  Prep   : "Imputed missing values (median/mode) to avoid losing rows; one-hot encoded
#            categoricals because models need numbers; scaled features because [KNN/SVM] are
#            distance-based; split 80/20 stratified; fit scaler on train only to prevent leakage."
#  Pipeline: "load -> clean -> encode -> split -> scale -> train -> predict -> evaluate (CV, tuning)."
#  Results: "Best model [X] achieved accuracy/F1/R2 = ...; confusion shows ...; improve with
#            cross-validation, tuning, more data, feature engineering."
#  Ethics : see bank above.
