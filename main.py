from src import *
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

MAIN_METRIC = calculate_accuracy_score

ada_boost_search_space = {
    "estimator": [DecisionTreeClassifier(max_depth=1),
                  DecisionTreeClassifier(max_depth=2)],
    "algorithm": ["SAMME"],
    "n_estimators": [10, 50, 100],
    "learning_rate": [1e-2, 1e-1, 2e-1]
}

decision_tree_search_space = {
    "splitter": ["best", "random"],
    "max_depth": [None, 10, 50, 100],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", "log2", None]
}

random_forest_search_space = {
    "n_estimators": [10, 50, 100],
    "max_depth": [10, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", "log2", None]
}

xgboost_search_space = {
    "eta": [1e-2, 1e-1, 2e-1],
    "max_depth": [3, 6, 9],
    "subsample": [0.5, 0.75, 1],
    "n_estimators": [10, 100, 200],
}

gradient_boosting_search_space = {
    "learning_rate": [1e-2, 1e-1, 2e-1],
    "max_depth": [3, 6, 9],
    "subsample": [0.5, 0.75, 1],
    "n_estimators": [10, 100, 200],
}

lightgbm_search_space = {
    "learning_rate": [1e-2, 1e-1, 2e-1],
    "max_depth": [3, 6, 9],
    "subsample": [0.5, 0.75, 1],
    "n_estimators": [10, 100, 200],
}


def main():
    problem_data = ProblemData("churn.csv")

    # CrossValidator is a class that takes a list of models, problem data, number of splits, and error metric.
    # It trains each model on the training data and evaluates it on the testing data using the error metric.
    # KFold is used to split the data into training and testing data.
    cross_validator = CrossValidator(
        model_names=[DecisionTree, AdaBoost, RandomForest, XGBoost, GradientBoosting, LightGBM],
        problem_data=problem_data,
        n_splits=5,
        error_metric=MAIN_METRIC
    )

    # The best model is selected based on the error metric.
    # The best model is tuned using hyper_search function.
    # GridSearch is a class that takes a model, problem data, search space, and error metric.
    # It searches for the best hyperparameters using grid search.
    # 5-fold cross validation is used to evaluate the model.
    params = hyper_search(cross_validator.best_model, problem_data)

    test_model(problem_data, LightGBM, params)


def test_model(problem_data: ProblemData, model: type, params: dict):
    model_instance = model(parameters=params)

    datasets = list(KFold(n_splits=5, shuffle=True, random_state=0)
                     .split(problem_data.df_X, problem_data.df_Y))

    auc_scores = []
    acc_scores = []
    for train_indices, test_indices in datasets:
        train_X, train_Y = (problem_data.df_X.values[train_indices],
                            problem_data.df_Y.values[train_indices])
        train_Y = train_Y.reshape(-1)

        test_X, test_Y = (problem_data.df_X.values[test_indices],
                          problem_data.df_Y.values[test_indices])
        test_Y = test_Y.reshape(-1)

        train_X, test_X = normalize(train_X, test_X)
        train_X, train_Y = oversample(train_X, train_Y)

        model_instance.fit(train_X, train_Y)

        auc_scores.append(calculate_auc_score(model_instance, test_X, test_Y))
        acc_scores.append(calculate_accuracy_score(model_instance, test_X, test_Y))

    print(f"Model: {model.__name__}")
    print(f"Params: {params}")
    print(f"AUC: {np.mean(auc_scores)}")
    print(f"Accuracy: {np.mean(acc_scores)}")

def hyper_search(model, problem_data) -> dict:
    if model == AdaBoost:
        search_space = ada_boost_search_space
    elif model == DecisionTree:
        search_space = decision_tree_search_space
    elif model == RandomForest:
        search_space = random_forest_search_space
    elif model == XGBoost:
        search_space = xgboost_search_space
    elif model == GradientBoosting:
        search_space = gradient_boosting_search_space
    elif model == LightGBM:
        search_space = lightgbm_search_space
    else:
        raise ValueError(f"Model {model.__name__} not supported")

    hyper_parameter_search = GridSearch(
        model=model,
        problem_data=problem_data,
        search_space=search_space,
        error_metric=MAIN_METRIC
    )

    return hyper_parameter_search.best_params[0]

# Normalization
# Scalers are trained of training data. Test data is normalized using trained scalers.
# For instance, when k-fold cross validation, a scaler is trained on training data and testing data
# then data is normalized using the scaler.
#
# Oversampling
# Likewise, for each fold, only the training data oversampled. Testing data is left untouched.
#
# Encoding
# While features that have ordinal structure encoded using OneHotEncoder, rest is encoded via LabelEncoder.


if __name__ == "__main__":
    main()
