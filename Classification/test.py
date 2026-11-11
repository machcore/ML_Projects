import main as mn
import pandas as pd


new_people = pd.DataFrame([
    [22.0,  20, "E-Reading",        15, True],
    [23.0,  45, "Messaging",        35, True],
    [23.5,  90, "Social Media",     65, False],
    [1.0,  180, "Gaming",           90, False],
    [0.5,  130, "Short-Form Video", 85, False],
    [22.5,  30, "News",             25, True],
    [2.0,  220, "Short-Form Video", 100, False],
    [0.0,  110, "Streaming",        70, False],
    [23.0,  60, "Gaming",           50, True],
    [1.5,  200, "Social Media",     95, False]
], columns=[
    "bedtime_hour",
    "screen_time_before_bed_min",
    "primary_app_category",
    "screen_brightness_pct",
    "blue_light_filter_on"
])

print(new_people)

X_new = mn.preprocess_new(new_people)

results = new_people.copy()

results["Histogram"] = mn.predict_histogram(X_new)
results["Gaussian"] = mn.predict_gaussian(X_new)
results["KDE"] = mn.predict_kde(X_new)
results["GMM"] = mn.predict_gmm(X_new)

