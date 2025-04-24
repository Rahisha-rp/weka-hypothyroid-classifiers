# WEKA Hypothyroid Classifiers Evaluation

This project evaluates various machine learning classifiers on the hypothyroid dataset using WEKA.

## Classifiers Implemented
- Naive Bayes
- J48 Decision Tree
- IBk (k-Nearest Neighbors)
- Logistic Regression
- Random Forest
- Voting Ensemble of all classifiers

## How to Run

1. Ensure you have Java installed
2. Download WEKA library (weka.jar)
3. Compile and run:
   ```bash
   javac -cp weka.jar src/HypothyroidClassifierEvaluation.java -d out/
   java -cp "out/;weka.jar" HypothyroidClassifierEvaluation
   ```

## Dataset
The hypothyroid dataset is included in the data/ directory.
