import numpy as np

from regularization import (
    get_regression_data,
    get_classification_data,
    linear_regression,
    ridge_regression,
    lasso_regression,
    logistic_regression,
    logistic_l2_regression,
    logistic_l1_regression,
)


def evaluate_regression_models():
    X_train, X_test, y_train, y_test = get_regression_data()
    models = {
        "Linear": linear_regression(X_train, y_train),
        "Ridge": ridge_regression(X_train, y_train),
        "Lasso": lasso_regression(X_train, y_train),
    }

    results = []
    for name, model in models.items():
        score = model.score(X_test, y_test)  # R2 score for regressors
        results.append((name, score))

    results.sort(key=lambda x: -x[1])

    print("Regression models (Diabetes dataset):")
    print("Model\tScore (R2)")
    for name, score in results:
        print(f"{name}\t{score:.4f}")


def evaluate_classification_models():
    X_train, X_test, y_train, y_test = get_classification_data()
    models = {
        "Logistic (no reg)": logistic_regression(X_train, y_train),
        "Logistic L2": logistic_l2_regression(X_train, y_train),
        "Logistic L1": logistic_l1_regression(X_train, y_train),
    }

    results = []
    for name, model in models.items():
        score = model.score(X_test, y_test)  # Accuracy for classifiers
        results.append((name, score))

    results.sort(key=lambda x: -x[1])

    print("\nClassification models (Breast Cancer dataset):")
    print("Model\tScore (Accuracy)")
    for name, score in results:
        print(f"{name}\t{score:.4f}")


def main():
    evaluate_regression_models()
    evaluate_classification_models()

if __name__ == "__main__":
    main()


