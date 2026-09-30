"""
Compare an entropy-based Decision Tree with unsmoothed Categorical Naive Bayes
on PlayTennis using Leave-One-Out Cross-Validation. Print probability tables
and evaluation metrics, then save comparison plots, confusion matrices,
and JSON results in outputs/.

ENCS3340 | Project Two | Lara Daifallah - 1230239
"""

import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("outputs", exist_ok=True)


# PlayTennis dataset from the slides
data = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"],
]

columns = ["Outlook", "Temperature", "Humidity", "Wind"]
X_raw = [row[:4] for row in data]
y_raw = [row[4] for row in data]


#convert text values into numbers because scklearn models need numeric input
encoders = [LabelEncoder() for _ in columns]
X_encoded = np.column_stack([
    enc.fit_transform([row[i] for row in X_raw])
    for i, enc in enumerate(encoders)
])

y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y_raw)


def build_nb_tables(X_raw, y_raw, columns):
    #build probability tables manually to match the Naive Bayes explanation in our slides
    priors = {c: y_raw.count(c) / len(y_raw) for c in ["Yes", "No"]}
    tables = {}

    for fi, feat in enumerate(columns):
        values = sorted(set(row[fi] for row in X_raw))
        tables[feat] = {}

        for val in values:
            tables[feat][val] = {}

            for cls in ["Yes", "No"]:
                count = sum(
                    1
                    for row, label in zip(X_raw, y_raw)
                    if row[fi] == val and label == cls
                )
                tables[feat][val][cls] = (count, y_raw.count(cls))

    return priors, tables


priors, nb_tables = build_nb_tables(X_raw, y_raw, columns)

print("=" * 65)
print("PART B -- Decision Tree vs Naive Bayes (PlayTennis)")
print("=" * 65)


# Print Naive Bayes probability tables
print("\n── Naive Bayes Probability Tables (full dataset, slide values) ──")
print(f"  P(Play=Yes) = {priors['Yes']:.4f}  (9/14)")
print(f"  P(Play=No)  = {priors['No']:.4f}  (5/14)\n")

for feat, vals in nb_tables.items():
    print(f"  {feat}:")
    print(f"    {'Value':<12} {'P(val|Yes)':<20} {'P(val|No)'}")
    print(f"    {'-' * 46}")

    for val, counts in vals.items():
        cy, ty = counts["Yes"]
        cn, tn = counts["No"]
        print(
            f"    {val:<12} {cy}/{ty} = {cy / ty:.4f}         "
            f"{cn}/{tn} = {cn / tn:.4f}"
        )

    print()


# Example calculation from slides
print("── NB Test Example ───────────")
print("  x' = (Outlook=Sunny, Temperature=Cool, Humidity=High, Wind=Strong)")

slide_yes = (9 / 14) * (2 / 9) * (3 / 9) * (3 / 9) * (3 / 9)
slide_no = (5 / 14) * (3 / 5) * (1 / 5) * (4 / 5) * (3 / 5)

print(f"\n  P(Yes|x') = (9/14)*(2/9)*(3/9)*(3/9)*(3/9) = {slide_yes:.4f}")
print(f"  P(No|x')  = (5/14)*(3/5)*(1/5)*(4/5)*(3/5) = {slide_no:.4f}")
print("  --> Prediction: No\n")


#Leave-One-Out Cross Validation is suitable cause the dataset is very small
loo = LeaveOneOut()

dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
nb = CategoricalNB(alpha=0.0, force_alpha=True)  # no Laplace smoothing so it matches our slides

dt_predictions = cross_val_predict(dt, X_encoded, y, cv=loo)
nb_predictions = cross_val_predict(nb, X_encoded, y, cv=loo)

dt_accuracy = accuracy_score(y, dt_predictions)
nb_accuracy = accuracy_score(y, nb_predictions)

dt_cm = confusion_matrix(y, dt_predictions)
nb_cm = confusion_matrix(y, nb_predictions)

dt_report = classification_report(
    y, dt_predictions, target_names=y_encoder.classes_,
    output_dict=True, zero_division=0
)

nb_report = classification_report(
    y, nb_predictions, target_names=y_encoder.classes_,
    output_dict=True, zero_division=0
)


print("=" * 65)
print("LOO Cross-Validation Results (14 folds, 1 test instance each)")
print("=" * 65)

print(f"\nDecision Tree : {dt_accuracy * 100:.1f}% ({int(dt_accuracy * 14)}/14 correct)")
print(classification_report(
    y, dt_predictions, target_names=y_encoder.classes_, zero_division=0
))

print(f"Naive Bayes   : {nb_accuracy * 100:.1f}% ({int(nb_accuracy * 14)}/14 correct)")
print(classification_report(
    y, nb_predictions, target_names=y_encoder.classes_, zero_division=0
))


#show prediction result for each instance
print("── Per-Instance LOO Predictions ──────────────────────────────────")
print(f"  {'#':>3}  {'Actual':<8}  {'DT Pred':<10}  {'DT':>4}  {'NB Pred':<10}  {'NB':>4}")
print("  " + "-" * 48)

for i in range(len(y)):
    actual = y_encoder.inverse_transform([y[i]])[0]
    dt_p = y_encoder.inverse_transform([dt_predictions[i]])[0]
    nb_p = y_encoder.inverse_transform([nb_predictions[i]])[0]

    print(
        f"  D{i+1:>2}  {actual:<8}  {dt_p:<10}  "
        f"{'OK' if dt_predictions[i] == y[i] else '--':>4}  "
        f"{nb_p:<10}  "
        f"{'OK' if nb_predictions[i] == y[i] else '--':>4}"
    )


#accuracy comparison chart
fig, ax = plt.subplots(figsize=(7, 5))

models = ["Decision Tree", "Naive Bayes"]
accuracies = [dt_accuracy * 100, nb_accuracy * 100]

bars = ax.bar(models, accuracies, width=0.45, edgecolor="white")

for bar, acc in zip(bars, accuracies):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=13,
    )

ax.set_ylim(0, 115)
ax.set_ylabel("LOO-CV Accuracy (%)")
ax.set_title("Decision Tree vs Naive Bayes\nLeave-One-Out CV Accuracy")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/part_b_comparison.png", dpi=150, bbox_inches="tight")
plt.close()


#confusion matrix plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, cm, title in zip(
    axes,
    [dt_cm, nb_cm],
    ["Decision Tree (LOO-CV)", "Naive Bayes (LOO-CV)"]
):
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(y_encoder.classes_)
    ax.set_yticklabels(y_encoder.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(
                c, r, str(cm[r, c]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

plt.suptitle("Confusion Matrices -- PlayTennis LOO-CV", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/part_b_confusion.png", dpi=150, bbox_inches="tight")
plt.close()


# Naive Bayes probability table figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, cls in enumerate(["Yes", "No"]):
    rows = []
    row_labels = []

    for feat in columns:
        for val, counts in nb_tables[feat].items():
            c, t = counts[cls]
            rows.append([f"{c}/{t}", f"{c/t:.3f}"])
            row_labels.append(f"{feat} = {val}")

    ax = axes[ax_idx]
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=["Count", f"P(.|Play={cls})"],
        loc="center",
        cellLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.3, 1.5)

    ax.set_title(f"P(feature | Play={cls})", fontweight="bold", fontsize=12, pad=14)

fig.suptitle("Naive Bayes Probability Tables (PlayTennis)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/part_b_nb_tables.png", dpi=150, bbox_inches="tight")
plt.close()


print("\n[saved] outputs/part_b_comparison.png")
print("[saved] outputs/part_b_confusion.png")
print("[saved] outputs/part_b_nb_tables.png")


# Save results for report 
results = {
    "evaluation": "Leave-One-Out Cross-Validation (14 folds)",
    "n_instances": len(data),
    "priors": {k: round(v, 4) for k, v in priors.items()},
    "slide_test_example": {
        "instance": "(Sunny, Cool, High, Strong)",
        "p_yes": round(slide_yes, 4),
        "p_no": round(slide_no, 4),
        "prediction": "No",
    },
    "decision_tree": {
        "accuracy": round(dt_accuracy * 100, 1),
        "correct": int(dt_accuracy * len(data)),
        "report": dt_report,
        "confusion": dt_cm.tolist(),
        "predictions": [
            y_encoder.inverse_transform([p])[0] for p in dt_predictions
        ],
        "actuals": [
            y_encoder.inverse_transform([a])[0] for a in y
        ],
    },
    "naive_bayes": {
        "accuracy": round(nb_accuracy * 100, 1),
        "correct": int(nb_accuracy * len(data)),
        "report": nb_report,
        "confusion": nb_cm.tolist(),
        "predictions": [
            y_encoder.inverse_transform([p])[0] for p in nb_predictions
        ],
        "actuals": [
            y_encoder.inverse_transform([a])[0] for a in y
        ],
    },
}

with open("outputs/part_b_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done.")