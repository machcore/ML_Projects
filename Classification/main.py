import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

df = pd.read_csv("Classification/sleep_debt_screen_time.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum())

features = [
    "bedtime_hour",
    "screen_time_before_bed_min",
    "primary_app_category",
    "screen_brightness_pct",
    "blue_light_filter_on"
]

target = "sleep_debt_category"

data = df[features + [target]].dropna().copy()

X = data[features]
y = data[target]

print(y.value_counts())
"""
sns.countplot(data=data, x=target)
plt.title("Sleep Debt Category Distribution")
plt.xticks(rotation=30)
plt.show()
"""

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


numeric_cols = [
    "bedtime_hour",
    "screen_time_before_bed_min",
    "screen_brightness_pct"
]

def encode(df):
    df = df.copy()

    df["blue_light_filter_on"] = (
        df["blue_light_filter_on"]
        .astype(str).str.lower().map({
            "true": 1, "false": 0,
            "1": 1, "0": 0
        })
    )

    df = pd.get_dummies(
        df,
        columns=["primary_app_category"],
        dtype=int
    )

    return df

train_encoded = encode(X_train_raw)
test_encoded = encode(X_test_raw)

test_encoded = test_encoded.reindex(
    columns=train_encoded.columns,
    fill_value=0
)

scaler = StandardScaler()

train_encoded[numeric_cols] = scaler.fit_transform(
    train_encoded[numeric_cols]
)

test_encoded[numeric_cols] = scaler.transform(
    test_encoded[numeric_cols]
)

X_train = train_encoded.to_numpy(dtype=float)
X_test = test_encoded.to_numpy(dtype=float)

classes = np.unique(y_train)
priors = {c: np.mean(y_train == c) for c in classes}

print("Number of features after encoding:", X_train.shape[1])
print("Classes:", classes)


def preprocess_new(df):
    encoded = encode(df)

    encoded = encoded.reindex(
        columns=train_encoded.columns,
        fill_value=0
    )

    encoded[numeric_cols] = scaler.transform(
        encoded[numeric_cols]
    )

    return encoded.to_numpy(dtype=float)


hist_models = {}
n_bins = 10

# Continuous features have histogram bins
num_indices = [
    train_encoded.columns.get_loc(c)
    for c in numeric_cols
]

# All remaining features are binary encoded variables
binary_indices = [
    j for j in range(X_train.shape[1])
    if j not in num_indices
]

# Use the same edges for every class
edges = {}

for j in num_indices:
    edges[j] = np.histogram_bin_edges(
        X_train[:, j], bins=n_bins
    )

for c in classes:
    X_c = X_train[y_train == c]
    hist_models[c] = {}

    for j in num_indices:
        counts, _ = np.histogram(
            X_c[:, j], bins=edges[j]
        )

        probs = (counts + 1) / (
            counts.sum() + len(counts)
        )

        hist_models[c][j] = probs

    for j in binary_indices:
        ones = np.sum(X_c[:, j] == 1)

        # Laplace smoothing
        p_one = (ones + 1) / (len(X_c) + 2)
        hist_models[c][j] = p_one


def predict_histogram(X):
    scores = []

    for c in classes:
        log_prob = np.full(len(X), np.log(priors[c]))

        for j in num_indices:
            idx = np.searchsorted(
                edges[j], X[:, j], side="right"
            ) - 1

            idx = np.clip(
                idx, 0, n_bins - 1
            )

            log_prob += np.log(
                hist_models[c][j][idx]
            )

        for j in binary_indices:
            p = hist_models[c][j]

            log_prob += np.where(
                X[:, j] == 1,
                np.log(p),
                np.log(1 - p)
            )

        scores.append(log_prob)

    return classes[np.argmax(scores, axis=0)]


y_pred_hist = predict_histogram(X_test)

print("Histogram accuracy:",
      accuracy_score(y_test, y_pred_hist))



gaussian_models = {}

for c in classes:
    X_c = X_train[y_train == c]

    mu = np.mean(X_c, axis=0)

    cov = np.cov(X_c, rowvar=False)

    # Regularization for stability
    cov += 0.2 * np.eye(X_train.shape[1])

    gaussian_models[c] = (mu, cov)



def predict_gaussian(X):
    scores = []

    for c in classes:
        mu, cov = gaussian_models[c]

        log_prob = multivariate_normal.logpdf(
            X, mean=mu, cov=cov
        )

        scores.append(log_prob + np.log(priors[c]))

    return classes[np.argmax(scores, axis=0)]


y_pred_gaussian = predict_gaussian(X_test)

print("Gaussian accuracy:",
      accuracy_score(y_test, y_pred_gaussian))



kde_models = {}

for c in classes:
    X_c = X_train[y_train == c]

    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=1.0
    )

    kde.fit(X_c)
    kde_models[c] = kde



def predict_kde(X):
    scores = []

    for c in classes:
        log_prob = kde_models[c].score_samples(X)

        scores.append(log_prob + np.log(priors[c]))

    return classes[np.argmax(scores, axis=0)]


y_pred_kde = predict_kde(X_test)

print("KDE accuracy:",
      accuracy_score(y_test, y_pred_kde))



gmm_models = {}

for c in classes:
    X_c = X_train[y_train == c]

    gmm = GaussianMixture(
        n_components=min(2, len(X_c)),
        covariance_type="diag",
        reg_covar=1e-3,
        random_state=42
    )

    gmm.fit(X_c)

    gmm_models[c] = gmm



def predict_gmm(X):
    scores = []

    for c in classes:
        log_prob = gmm_models[c].score_samples(X)

        scores.append(log_prob + np.log(priors[c]))

    return classes[np.argmax(scores, axis=0)]


y_pred_gmm = predict_gmm(X_test)

print("GMM accuracy:",
      accuracy_score(y_test, y_pred_gmm))



from matplotlib.colors import ListedColormap

predictors = {
    "Histogram": predict_histogram,
    "Gaussian": predict_gaussian,
    "KDE": predict_kde,
    "GMM": predict_gmm
}

colors = ["#3A9D79", "#E0AE46", "#EC8147", "#B25278"]

# Use a consistent class order
class_order = [
    c for c in ["No Debt", "None", "Mild", "Moderate", "Severe"]
    if c in classes
]

# Include any other labels if necessary
class_order += [c for c in classes if c not in class_order]

class_colors = colors[:len(class_order)]
cmap = ListedColormap(class_colors)

# Representative observation
reference = X_train_raw.iloc[[0]].copy()

reference["bedtime_hour"] = X_train_raw["bedtime_hour"].median()
reference["primary_app_category"] = (
    X_train_raw["primary_app_category"].mode()[0]
)
reference["blue_light_filter_on"] = (
    X_train_raw["blue_light_filter_on"].mode()[0]
)

# Create a grid for screen time and brightness
x_min = X_train_raw["screen_time_before_bed_min"].min()
x_max = X_train_raw["screen_time_before_bed_min"].max()

y_min = X_train_raw["screen_brightness_pct"].min()
y_max = X_train_raw["screen_brightness_pct"].max()

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

grid = pd.concat(
    [reference] * xx.size,
    ignore_index=True
)

grid["screen_time_before_bed_min"] = xx.ravel()
grid["screen_brightness_pct"] = yy.ravel()

grid_processed = preprocess_new(grid)



"""
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

for ax, (name, predict) in zip(
    axes.flat, predictors.items()
):
    # Predict on the 2D grid using all features
    predictions = predict(grid_processed)

    Z = np.array([
        class_order.index(c) for c in predictions
    ]).reshape(xx.shape)

    ax.contourf(
        xx, yy, Z,
        levels=np.arange(len(class_order) + 1) - 0.5,
        cmap=cmap,
        alpha=0.35
    )

    # Show actual test observations
    for i, c in enumerate(class_order):
        mask = y_test == c

        ax.scatter(
            X_test_raw.loc[mask, "screen_time_before_bed_min"],
            X_test_raw.loc[mask, "screen_brightness_pct"],
            color=class_colors[i],
            s=12,
            alpha=0.5,
            label=c
        )

    ax.set_title(name)
    ax.set_xlabel("Screen Time Before Bed (min)")
    ax.set_ylabel("Screen Brightness (%)")
    ax.legend()

plt.suptitle(
    "Bayesian Classification: Decision Boundaries",
    fontsize=16
)

plt.tight_layout()
plt.show()
"""


predictions = {
    "Histogram": y_pred_hist,
    "Gaussian": y_pred_gaussian,
    "KDE": y_pred_kde,
    "GMM": y_pred_gmm
}

"""
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

for ax, (name, pred) in zip(
    axes.flat, predictions.items()
):
    cm = confusion_matrix(
        y_test, pred, labels=class_order
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_order,
        yticklabels=class_order,
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

plt.tight_layout()
plt.show()
"""


results = []

for name, pred in predictions.items():
    results.append({
        "Method": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Macro F1": f1_score(
            y_test, pred, average="macro", zero_division=0
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, pred
        )
    })

results_df = pd.DataFrame(results)

print(results_df.round(3))


"""
results_df.set_index("Method").plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Performance Comparison of Classification Methods")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

"""

from scipy.stats import multivariate_normal

def gaussian_pdf(X, c):
    mu, cov = gaussian_models[c]

    return multivariate_normal.pdf(
        X,
        mean=mu,
        cov=cov
    )


def kde_pdf(X, c):
    return np.exp(
        kde_models[c].score_samples(X)
    )

def gmm_pdf(X, c):
    return np.exp(
        gmm_models[c].score_samples(X)
    )


def histogram_pdf(X, c):
    density = np.ones(len(X))

    for j in num_indices:
        bin_edges = edges[j]

        idx = np.searchsorted(
            bin_edges, X[:, j], side="right"
        ) - 1

        idx = np.clip(idx, 0, n_bins - 1)

        probs = hist_models[c][j][idx]
        widths = np.diff(bin_edges)[idx]

        density *= probs / widths

    for j in binary_indices:
        p = hist_models[c][j]

        density *= np.where(
            X[:, j] == 1,
            p,
            1 - p
        )

    return density

screen_times = np.linspace(5, 240, 200)

sample = pd.DataFrame({
    "bedtime_hour": 22.0,
    "screen_time_before_bed_min": screen_times,
    "primary_app_category": "Gaming",
    "screen_brightness_pct": 10,
    "blue_light_filter_on": True
})

X_grid = preprocess_new(sample)

pdf_methods = {
    "Histogram": histogram_pdf,
    "Gaussian": gaussian_pdf,
    "KDE": kde_pdf,
    "GMM": gmm_pdf
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, (name, pdf_func) in zip(
    axes.flat, pdf_methods.items()
):
    for c in classes:
        values = pdf_func(X_grid, c)

        ax.plot(
            screen_times,
            values,
            label=c
        )

    ax.set_title(name)
    ax.set_xlabel("Screen Time Before Bed (minutes)")
    ax.set_ylabel("Conditional density")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()