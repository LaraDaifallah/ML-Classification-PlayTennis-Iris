# ML Classification: PlayTennis + Iris

ID3 from scratch and Decision Tree vs. Naive Bayes on PlayTennis, plus Decision Tree classification on the Iris dataset.

## Experiments

| Part | Dataset | Approach | Evaluation |
| --- | --- | --- | --- |
| A | PlayTennis (14 examples) | ID3 implemented from scratch with entropy, information gain, and multi-way splits | Training predictions and tree visualization |
| B | PlayTennis (14 examples) | Scikit-learn Decision Tree (entropy) vs. Categorical Naive Bayes | Leave-One-Out Cross-Validation |
| C | Iris (150 samples, 3 species) | Decision Tree with Gini impurity and maximum depth 4 | Stratified 80/20 train/test split, confusion matrix, precision, recall, and F1-score |

## Run

From the repository root, install the dependencies and run each experiment:

```bash
python -m pip install -r requirements.txt
python src/playtennis_id3_from_scratch.py
python src/playtennis_decision_tree_vs_naive_bayes.py
python src/iris_decision_tree_classification.py
```

The PlayTennis data is embedded in the scripts. Iris is loaded using scikit-learn, so no separate dataset download is required. Figures and JSON results are saved in `outputs/` relative to the current working directory. Part A also attempts to open its saved figures in the system image viewer; on a headless machine, view the PNG files manually.

## Reported results

The submitted report records 100% training accuracy for the ID3 tree, 57.1% Leave-One-Out accuracy for both classifiers in Part B, and 93.3% test accuracy for Iris in Part C. Training accuracy and cross-validation/test accuracy measure different things and should not be compared directly.

Part B deliberately uses `alpha=0` for unsmoothed Naive Bayes to match the course calculations. Zero probabilities can affect predictions; behavior may vary across scikit-learn versions. These figures are the submitted results, not a guarantee for every dependency version.

## Files

- `src/playtennis_id3_from_scratch.py`: ID3 construction, entropy and information gain analysis, and tree plots.
- `src/playtennis_decision_tree_vs_naive_bayes.py`: classifier comparison, probability tables, and confusion matrices.
- `src/iris_decision_tree_classification.py`: Iris classification, evaluation metrics, and feature importance plots.
- [Project report](docs/project-report.pdf): methodology, figures, results, and discussion.

## Author

Lara Daifallah — Birzeit University, ENCS3340 Artificial Intelligence, Project Two (2025–2026).
