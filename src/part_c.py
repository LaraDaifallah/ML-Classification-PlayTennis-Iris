"""
Part C : Classification on the UCI Iris Dataset
ENCS3340 || Project 1 || Lara Daifallah - 1230239
"""

import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# Folder for saving figures and results
os.makedirs("outputs", exist_ok=True)


# Load the Iris dataset from sklearn
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
class_names = list(iris.target_names)


print("=" * 60)
print("PART C - Decision Tree on UCI Iris Dataset")
print("=" * 60)

print("\nDataset      : UCI Iris")
print(f"Instances    : {X.shape[0]}")
print(f"Features     : {X.shape[1]}  {list(feature_names)}")
print(f"Classes      : {class_names}")


#simple feature statistics to describe the dataset
print("\nFeature statistics (mean ± std):")
for i, name in enumerate(feature_names):
    print(
        f"  {name:<25}: {X[:, i].mean():.2f} ± {X[:, i].std():.2f}  "
        f"[{X[:, i].min():.1f} - {X[:, i].max():.1f}]"
    )


#split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,  #stratify keeps the class distribution balanced in both sets

)

print(f"\nTraining set : {len(X_train)} instances")
print(f"Testing set  : {len(X_test)} instances")


#train a Decision Tree classifier
# max_depth=4 is used to keep the tree simple and reduce overfitting
clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42,
)

clf.fit(X_train, y_train)


#test the model on unseen testing data
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True,
)

report_str = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
)


print(f"\nTest Accuracy : {accuracy * 100:.1f}%\n")

print("Classification Report:")
print(report_str)

print("Confusion Matrix:")
print(cm)


#figure 1: Decision Tree
fig, ax = plt.subplots(figsize=(18, 9))

plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax,
    impurity=True,
    precision=2,
)

ax.set_title(
    "Decision Tree - UCI Iris Dataset",
    fontsize=14,
    fontweight="bold",
    pad=16,
)

plt.tight_layout()
plt.savefig("outputs/part_c_tree.png", dpi=150, bbox_inches="tight")
plt.close()

print("[saved] outputs/part_c_tree.png")


#figure 2: Confusion Matrix 
fig, ax = plt.subplots(figsize=(6, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

disp.plot(ax=ax, colorbar=False, cmap="Blues")

ax.set_title(
    "Confusion Matrix - Iris Test Set",
    fontsize=13,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig("outputs/part_c_confusion.png", dpi=150, bbox_inches="tight")
plt.close()

print("[saved] outputs/part_c_confusion.png")


#figure 3: Precision,Recall,and F1-score 
metrics = ["precision", "recall", "f1-score"]

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))

for i, metric in enumerate(metrics):
    values = [report[cls][metric] for cls in class_names]

    bars = ax.bar(
        x + i * width,
        values,
        width,
        label=metric.capitalize(),
        edgecolor="white",
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

ax.set_xticks(x + width)
ax.set_xticklabels(class_names, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)

ax.set_title(
    "Precision, Recall, and F1-score by Class - Iris Test Set",
    fontsize=12,
    fontweight="bold",
)

ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/part_c_metrics.png", dpi=150, bbox_inches="tight")
plt.close()

print("[saved] outputs/part_c_metrics.png")


# figure 4: Feature Importances 
# This shows which features were most useful for the Decision Tree.
importances = clf.feature_importances_

fig, ax = plt.subplots(figsize=(8, 4))

bars = ax.barh(feature_names, importances)

ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
ax.set_title(
    "Feature Importances - Iris Decision Tree",
    fontsize=12,
    fontweight="bold",
)

for bar, importance in zip(bars, importances):
    ax.text(
        importance + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{importance:.3f}",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/part_c_importances.png", dpi=150, bbox_inches="tight")
plt.close()

print("[saved] outputs/part_c_importances.png")


# Save important results for the report
results = {
    "dataset": "UCI Iris",
    "n_instances": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "feature_names": list(feature_names),
    "class_names": class_names,
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "accuracy": round(accuracy * 100, 1),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "feature_importances": {
        name: round(float(importance), 4)
        for name, importance in zip(feature_names, importances)
    },
}

with open("outputs/part_c_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone.")