import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class HypothyroidClassifierEvaluation {

    public static void main(String[] args) {
        try {
            // Load the hypothyroid dataset
            DataSource source = new DataSource("hypothyroid.arff");
            Instances data = source.getDataSet();
            
            // Set the class index to the last attribute
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            
            // Initialize classifiers
            Classifier[] classifiers = {
                new NaiveBayes(),       // Naive Bayes
                new J48(),               // J48 Decision Tree
                new IBk(3),              // IBk (kNN) with k=3
                new Logistic(),          // Logistic Regression
                new RandomForest()      // Random Forest
            };
            
            String[] classifierNames = {
                "Naive Bayes",
                "J48",
                "IBk (k=3)",
                "Logistic Regression",
                "Random Forest"
            };
            
            // Evaluate individual classifiers
            System.out.println("Individual Classifier Performance:");
            System.out.println("==================================");
            
            for (int i = 0; i < classifiers.length; i++) {
                evaluateClassifier(classifiers[i], classifierNames[i], data);
            }
            
            // Evaluate combination of classifiers using Vote
            System.out.println("\nCombined Classifier Performance (Voting):");
            System.out.println("=========================================");
            
            Vote votingClassifier = new Vote();
            votingClassifier.setClassifiers(classifiers);
            evaluateClassifier(votingClassifier, "Voting Classifier", data);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    private static void evaluateClassifier(Classifier classifier, String classifierName, Instances data) throws Exception {
        // Perform 10-fold cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        
        // Print evaluation results
        System.out.println("\nClassifier: " + classifierName);
        System.out.println("----------------------------------------");
        System.out.println("Accuracy: " + String.format("%.2f%%", eval.pctCorrect()));
        System.out.println("Precision (Weighted): " + String.format("%.2f", eval.weightedPrecision()));
        System.out.println("Recall (Weighted): " + String.format("%.2f", eval.weightedRecall()));
        System.out.println("F-Measure (Weighted): " + String.format("%.2f", eval.weightedFMeasure()));
        System.out.println("AUC: " + String.format("%.2f", eval.weightedAreaUnderROC()));
        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());
    }
}