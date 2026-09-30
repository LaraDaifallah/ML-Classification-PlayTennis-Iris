"""
Build a multi-way ID3 decision tree from scratch on the 14-example PlayTennis dataset.
Compute entropy and information gain, evaluate training predictions, and save
the decision tree, information-gain chart, and JSON results in outputs/.

ENCS3340 | Project Two | Lara Daifallah - 1230239
"""

import math
import json
import os
import platform
import subprocess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#folder for saving figures and results
os.makedirs("outputs", exist_ok=True)


def open_file(path):
    #opens the saved image 
    path = os.path.abspath(path)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.call(["open", path])
    else:
        subprocess.call(["xdg-open", path])


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

attributes = ["Outlook", "Temperature", "Humidity", "Wind"]


def entropy(labels):
    #entropy measurement
    total = len(labels)
    counts = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent


def majority_class(rows):
    #used only if the algorithm reaches a stop case
    labels = [row[-1] for row in rows]
    counts = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    return max(counts, key=counts.get)


def information_gain(rows, attr_index):
    # IG = entropy before split - weighted entropy after split
    parent_labels = [row[-1] for row in rows]
    parent_entropy = entropy(parent_labels)

    values = sorted(set(row[attr_index] for row in rows))
    weighted_entropy = 0.0

    for value in values:
        subset = [row for row in rows if row[attr_index] == value]
        subset_labels = [row[-1] for row in subset]
        weighted_entropy += (len(subset) / len(rows)) * entropy(subset_labels)

    return parent_entropy - weighted_entropy


def get_split_details(rows, attr_index):
    #details printed to show how each attribute splits the dataset
    values = sorted(set(row[attr_index] for row in rows))
    details = {}

    for value in values:
        subset = [row for row in rows if row[attr_index] == value]
        labels = [row[-1] for row in subset]

        details[value] = {
            "Yes": labels.count("Yes"),
            "No": labels.count("No"),
            "Entropy": entropy(labels),
            "Size": len(subset),
        }

    return details


def build_id3_tree(rows, available_attrs):
    #build the tree by choosing the attribute with highest IG
    labels = [row[-1] for row in rows]

    if len(set(labels)) == 1:
        return {
            "type": "leaf",
            "class": labels[0],
            "samples": len(rows),
            "yes": labels.count("Yes"),
            "no": labels.count("No"),
            "entropy": entropy(labels),
        }

    if not available_attrs:
        maj = majority_class(rows)
        return {
            "type": "leaf",
            "class": maj,
            "samples": len(rows),
            "yes": labels.count("Yes"),
            "no": labels.count("No"),
            "entropy": entropy(labels),
        }

    gains = {
        attr: information_gain(rows, attributes.index(attr))
        for attr in available_attrs
    }

    best_attr = max(gains, key=gains.get)
    best_index = attributes.index(best_attr)

    node = {
        "type": "node",
        "attribute": best_attr,
        "ig": gains[best_attr],
        "samples": len(rows),
        "yes": labels.count("Yes"),
        "no": labels.count("No"),
        "entropy": entropy(labels),
        "children": {},
    }

    for value in sorted(set(row[best_index] for row in rows)):
        subset = [row for row in rows if row[best_index] == value]
        remaining_attrs = [a for a in available_attrs if a != best_attr]
        node["children"][value] = build_id3_tree(subset, remaining_attrs)

    return node


def predict(tree, instance):
    if tree["type"] == "leaf":
        return tree["class"]

    attr_index = attributes.index(tree["attribute"])
    value = instance[attr_index]

    if value in tree["children"]:
        return predict(tree["children"][value], instance)

    return "Yes"


def print_tree(tree, indent=""):
    if tree["type"] == "leaf":
        print(indent + f"→ {tree['class']} "
              f"(samples={tree['samples']}, Yes={tree['yes']}, No={tree['no']})")
        return

    print(indent + f"{tree['attribute']} "
          f"(IG={tree['ig']:.3f}, Entropy={tree['entropy']:.3f})")

    for value, child in tree["children"].items():
        print(indent + f"  [{value}]")
        print_tree(child, indent + "     ")


def assign_positions(tree, depth=0, x_left=0.0, x_right=1.0, positions=None):
    if positions is None:
        positions = {}

    x = (x_left + x_right) / 2
    positions[id(tree)] = (x, -depth)

    if tree["type"] == "node":
        children = list(tree["children"].values())
        n = len(children)

        for i, child in enumerate(children):
            child_left = x_left + (x_right - x_left) * i / n
            child_right = x_left + (x_right - x_left) * (i + 1) / n
            assign_positions(child, depth + 1, child_left, child_right, positions)

    return positions


def plot_id3_tree(tree, path):
    #plot the ID3 tree
    positions = assign_positions(tree)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    def draw(node):
        x, y = positions[id(node)]

        if node["type"] == "leaf":
            color = "lightgreen" if node["class"] == "Yes" else "mistyrose"
            text = f"{node['class']}\nYes={node['yes']}, No={node['no']}\nEntropy={node['entropy']:.3f}"
        else:
            color = "lightblue"
            text = f"{node['attribute']}\nIG={node['ig']:.3f}\nEntropy={node['entropy']:.3f}"

        ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor="black"),
        )

        if node["type"] == "node":
            for edge_value, child in node["children"].items():
                child_x, child_y = positions[id(child)]

                ax.annotate(
                    "",
                    xy=(child_x, child_y + 0.18),
                    xytext=(x, y - 0.18),
                    arrowprops=dict(arrowstyle="->", lw=1.2),
                )

                ax.text(
                    (x + child_x) / 2,
                    (y + child_y) / 2 + 0.08,
                    edge_value,
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

                draw(child)

    draw(tree)

    ys = [pos[1] for pos in positions.values()]
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(min(ys) - 0.6, 0.6)

    ax.set_title("ID3 Decision Tree - PlayTennis Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


print("=" * 60)
print("PART A - ID3 DECISION TREE on PlayTennis Dataset")
print("=" * 60)

labels = [row[-1] for row in data]
total_entropy = entropy(labels)

print(f"\nDataset: {len(data)} instances | Yes={labels.count('Yes')} No={labels.count('No')}")
print(f"Total Entropy = {total_entropy:.3f} bits\n")

print("Information Gain and Entropy for each Attribute")
ig_results = {}

for attr in attributes:
    attr_index = attributes.index(attr)
    ig = information_gain(data, attr_index)
    details = get_split_details(data, attr_index)

    ig_results[attr] = {"ig": ig, "details": details}

    print(f"\nGain(S, {attr}) = {ig:.3f}")

    for value, d in details.items():
        print(
            f"   {value:<10}: Yes={d['Yes']}, No={d['No']}, "
            f"({d['Size']}/{len(data)}), Entropy={d['Entropy']:.3f}"
        )

best_attr = max(ig_results, key=lambda a: ig_results[a]["ig"])
print(f"\nRoot attribute: {best_attr} (highest IG = {ig_results[best_attr]['ig']:.3f})")

#build and display the learned tree
tree = build_id3_tree(data, attributes)

print("\nLearned ID3 Tree")
print_tree(tree)

predictions = [predict(tree, row[:4]) for row in data]
correct = sum(1 for pred, row in zip(predictions, data) if pred == row[-1])
accuracy = correct / len(data)

print(f"\nTraining Accuracy: {accuracy * 100:.1f}%")

print("\nPredictions vs Actual")
for i, (pred, row) in enumerate(zip(predictions, data)):
    actual = row[-1]
    status = "OK" if pred == actual else "WRONG"
    print(f"  D{i+1:>2}: Predicted={pred}, Actual={actual} [{status}]")


#information gain bar chart
plt.figure(figsize=(8, 4))

attrs = list(ig_results.keys())
igs = [ig_results[attr]["ig"] for attr in attrs]

bars = plt.bar(attrs, igs)

for bar, val in zip(bars, igs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val:.3f}",
        ha="center",
        va="bottom",
    )

plt.ylabel("Information Gain (bits)")
plt.title("Information Gain per Attribute - PlayTennis")
plt.ylim(0, max(igs) * 1.25)
plt.tight_layout()

ig_path = os.path.abspath("outputs/part_a_ig.png")
plt.savefig(ig_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[saved] {ig_path}")


#ID3 tree plot
tree_path = os.path.abspath("outputs/part_a_id3_tree.png")
plot_id3_tree(tree, tree_path)

print(f"[saved] {tree_path}")

#open both saved figures
open_file(tree_path)
open_file(ig_path)


results = {
    "dataset_size": len(data),
    "yes_count": labels.count("Yes"),
    "no_count": labels.count("No"),
    "total_entropy": round(total_entropy, 3),
    "information_gain": {
        attr: round(ig_results[attr]["ig"], 3) for attr in attributes
    },
    "root_attribute": best_attr,
    "training_accuracy": round(accuracy * 100, 1),
    "predictions": [
        {
            "instance": f"D{i+1}",
            "predicted": predictions[i],
            "actual": data[i][-1],
            "correct": predictions[i] == data[i][-1],
        }
        for i in range(len(data))
    ],
}

with open("outputs/part_a_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done.")