""" Cameron Ela
    Description: This program trains and tests an SVM model
    based on a dataset regarding white wine quality.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


# cleans data of duplicates and nulls
def clean(df):
    new_df = df.dropna()
    new_df = new_df.drop_duplicates()

    return new_df


# transforms feature vector by standardizing it
def standardize(X):
    tf = StandardScaler()
    X_tf = tf.fit_transform(X)
    X_tf = pd.DataFrame(X_tf, columns=X.columns)

    return X_tf


""" returns highest three correlated attributes in descending order given
    a feature vector, target, and the column name of the output/target
"""


def find_corr_attr(feature, target, output):
    combined = feature
    combined[output] = target
    corr_matrix = combined.corr()
    # adds correlations between attributes and outputs to list
    corr = []  # list holding highest correlations
    for name in corr_matrix.columns:
        if name != output:  # avoids output self-correlation
            corr.append(abs(corr_matrix[name][output]))  # abs for sorting later
    corr.sort(reverse=True)  # sorts by descending absolute value

    # creates list of attributes based on correlation
    corr_name = []  # list holding highest correlated attributes
    for i in range(3):
        for name in corr_matrix.columns:
            # matches highest absolute correlations to column names
            if corr[i] == abs(corr_matrix[name][output]):
                corr_name.append(name)
                break

    return corr_name


# cross validation on SVM
def cross_validate(X_train, y_train):
    model = SVC()
    C = [0.1, 1, 10, 100]
    gamma = [0.2, 2, 20, 200]
    # SVM hyperparameters
    param_grid = {
        "C": C,
        "kernel": ["linear", "rbf"],
        "gamma": gamma
    }
    gscv = GridSearchCV(model, param_grid, verbose=3)
    # cross validation process
    gscv.fit(X_train, y_train)
    best = gscv.best_params_
    # creating and fitting optimal model
    optimized_model = SVC(C=best["C"], kernel=best["kernel"],
                          gamma=best["gamma"])
    optimized_model.fit(X_train, y_train)

    return optimized_model, gscv


def main():
    # wrangles data from csv
    wine = pd.read_csv("winequality-white.csv", skiprows=1, sep=";")
    wine = clean(wine)
    target = wine["quality"]
    feature = wine.drop(columns="quality")
    feature_tf = standardize(feature)
    # get 3 highest-correlated attributes
    corr_attr = find_corr_attr(feature_tf, target, "quality")
    # drop columns that are the ouput or are not most statistically correlated
    for col in feature_tf.columns:
        if col not in corr_attr:
            feature_tf = feature_tf.drop(columns=col)

    # partition data
    X_train, X_test, y_train, y_test = train_test_split(feature_tf, target,
                                                        random_state=42, stratify=target)
    # cross validation (gscv needed for plotting accuracy scores)
    trained_model, gscv = cross_validate(X_train, y_train)

    # creating and plotting figure
    fig, ax = plt.subplots(1, 2, figsize=(20, 9))
    hyperparam_index = np.arange(len(gscv.cv_results_["mean_test_score"]))
    ax[0].plot(hyperparam_index, gscv.cv_results_["mean_test_score"])
    ax[0].set(xlabel="Hyperparameter Setup Index", ylabel="Acurracy Score",
              title=("Hyperparameter Setup vs. Accuracy Score\n"
                     "Highest Score={:.2f}".format(gscv.best_score_)))
    # plotting confusion matrix
    y_pred = trained_model.predict(X_test)
    mat = confusion_matrix(y_test, y_pred)
    mat_disp = ConfusionMatrixDisplay(confusion_matrix=mat)
    mat_disp.plot(ax=ax[1])
    ax[1].set(title="White Wine Quality Classification Confusion Matrix")
    fig.suptitle("White Wine Quality SVM Classification Results")
    fig.tight_layout()
    plt.savefig("winequality.png")


if __name__ == '__main__':
    main()
