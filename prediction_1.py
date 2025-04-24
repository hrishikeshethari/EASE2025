# import numpy as np
# import pandas as pd
# from scipy.stats import friedmanchisquare
# from scikit_posthocs import posthoc_nemenyi_friedman
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import MinMaxScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import roc_auc_score
#
#
# # Function to preprocess data
# def preprocess_data(train_file, test_file, feature_combinations):
#     train_df = pd.read_csv(train_file)
#     test_df = pd.read_csv(test_file)
#
#     # Convert Bugs to binary
#     train_df["Bugs"] = (train_df["Bugs"] > 0).astype(int)
#     test_df["Bugs"] = (test_df["Bugs"] > 0).astype(int)
#
#     results = {}
#     for i, features in enumerate(feature_combinations, 1):
#         X_train = train_df[list(features)]
#         X_test = test_df[list(features)]
#
#         # MinMax Scaling
#         scaler = MinMaxScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#
#         # Apply SMOTE
#         smote = SMOTE(random_state=42)
#         X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, train_df["Bugs"])
#
#         results[f"combination_{i}"] = (X_train_resampled, y_train_resampled, X_test_scaled, test_df["Bugs"])
#
#     return results
#
#
# # Train models and get predictions with AUROC calculation
# def train_and_predict(model, model_name, data):
#     predictions = {}
#     aurocs = {}
#
#     for i, (key, (X_train, y_train, X_test, y_test)) in enumerate(data.items(), 1):
#         clf = model
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#
#         # Compute AUROC
#         if hasattr(clf, "predict_proba"):
#             y_proba = clf.predict_proba(X_test)[:, 1]
#         else:
#             y_proba = clf.decision_function(X_test)
#
#         auroc = roc_auc_score(y_test, y_proba)
#
#         predictions[key] = y_pred
#         aurocs[key] = auroc
#         print(key, auroc)
#
#     return predictions, aurocs
#
#
# # Function to perform Friedman and Nemenyi tests
# def friedman_nemenyi_analysis(aurocs, project, model_name):
#     results = []
#
#     keys = list(aurocs.keys())
#     auroc_values = [np.atleast_1d(aurocs[key]) for key in keys]  # Convert scalars to arrays
#     print(auroc_values)
#
#     # Ensure each combination has the same number of AUROC values
#     min_length = min(len(a) for a in auroc_values)
#     auroc_values = [a[:min_length] for a in auroc_values]  # Truncate to match shortest
#     print(*auroc_values)
#
#     print(f"\nPerforming Friedman Test for {model_name} on {project}")
#
#     # Perform Friedman Test
#     friedman_stat, p_value = friedmanchisquare(*auroc_values)
#
#     results.append({
#         "project": project,
#         "model": model_name,
#         "test": "Friedman",
#         "p_value": p_value
#     })
#
#     print(f"Friedman Test p-value: {p_value}, stat = {friedman_stat}")
#
#     if p_value < 0.05:
#         print("Significant difference found, performing Nemenyi post-hoc test...")
#
#         # Convert AUROC values into a matrix format
#         auroc_matrix = np.array(auroc_values).T
#
#         # Perform Nemenyi Test
#         nemenyi_results = posthoc_nemenyi_friedman(auroc_matrix)
#
#         for i in range(len(keys)):
#             for j in range(i + 1, len(keys)):
#                 results.append({
#                     "project": project,
#                     "model": model_name,
#                     "test": "Nemenyi",
#                     "combination": f"{keys[i]} vs {keys[j]}",
#                     "p_value": nemenyi_results.iloc[i, j]
#                 })
#
#         print(nemenyi_results)
#
#     return results
#
#
# # Main execution function
# def main(project, feature_combinations, all_results):
#     train_file = f"{project}_training_file_entropy_bugs_bug_priorities_non_fatty_h1_final.csv"
#     test_file = f"{project}_test_file_entropy_bugs_bug_priorities_non_fatty_h1_final.csv"
#     print(f"\nProcessing project: {project}")
#
#     # Process data
#     data = preprocess_data(train_file, test_file, feature_combinations)
#
#     classifiers = {
#         'LR': LogisticRegression(random_state=42),
#         'SVM': SVC(probability=True, random_state=42),
#         'RF': RandomForestClassifier(random_state=42),
#         'DT': DecisionTreeClassifier(random_state=42),
#         'NB': GaussianNB(),
#         'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
#         'GB': GradientBoostingClassifier(random_state=42)
#     }
#
#     for model_name, clf in classifiers.items():
#         print(f"\nRunning {model_name} classifier...")
#         predictions, aurocs = train_and_predict(clf, model_name, data)
#         all_results.extend(friedman_nemenyi_analysis(aurocs, project, model_name))
#
#
# # Example usage
# feature_combinations = [
#     ['comm', 'adev', 'ddev', 'add', 'del', 'own', 'minor', 'ncomm', 'nadev',
#      'nddev', 'oexp', 'exp', "Change_entropy"],
#     ['comm', 'adev', 'ddev', 'add', 'del', 'own', 'minor', 'ncomm', 'nadev',
#      'nddev', 'oexp', 'exp', "Co_change_entropy"],
#     ['comm', 'adev', 'ddev', 'add', 'del', 'own', 'minor', 'ncomm', 'nadev',
#      'nddev', 'oexp', 'exp', "Change_entropy", "Co_change_entropy"]
# ]
#
# projects = ["derby", "activemq", "pdfbox", "pig", "kafka", "maven", "struts", "nifi"]
# all_results = []
#
# for project in projects:
#     main(project, feature_combinations, all_results)
#
# # Save all results in a single CSV file
# results_df = pd.DataFrame(all_results)
# results_df.to_csv("all_projects_entropy_friedman_nemenyi_results.csv", index=False)
#
# print("\n✅ Analysis complete! Results saved to 'friedman_nemenyi_results.csv'.")


import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_conover_friedman, posthoc_nemenyi_friedman
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


# Function to preprocess data
def preprocess_data(train_file, test_file, feature_combinations):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Convert Bugs to binary
    train_df["Bugs"] = (train_df["Bugs"] > 0).astype(int)
    test_df["Bugs"] = (test_df["Bugs"] > 0).astype(int)

    results = {}
    for i, features in enumerate(feature_combinations, 1):
        X_train = train_df[list(features)]
        X_test = test_df[list(features)]

        # MinMax Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, train_df["Bugs"])

        results[f"combination_{i}"] = (X_train_resampled, y_train_resampled, X_test_scaled, test_df["Bugs"])

    return results


# Train models and get predictions with multiple metrics
def train_and_predict(model, data):
    metrics = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1_score": {},
        "mcc": {},
        "auroc": {}
    }

    for key, (X_train, y_train, X_test, y_test) in data.items():
        clf = model
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Compute probability scores for AUROC
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_proba = clf.decision_function(X_test)

        # Compute Metrics
        metrics["accuracy"][key] = accuracy_score(y_test, y_pred)
        metrics["precision"][key] = precision_score(y_test, y_pred, zero_division=0)
        metrics["recall"][key] = recall_score(y_test, y_pred, zero_division=0)
        metrics["f1_score"][key] = f1_score(y_test, y_pred, zero_division=0)
        metrics["mcc"][key] = matthews_corrcoef(y_test, y_pred)
        metrics["auroc"][key] = roc_auc_score(y_test, y_proba)

    return metrics


# Perform Friedman & Post-Hoc tests on the combined matrix
def perform_statistical_tests(all_metrics, metric_name):
    df_matrix = pd.DataFrame(all_metrics, columns=["combination_1", "combination_2", "combination_3"])
    print(f"\n{metric_name.upper()} Matrix:\n", df_matrix)
    mean_values = df_matrix.mean()
    print(f"\nMean values for {metric_name}:")
    print(mean_values)

    # Friedman Test
    friedman_stat, p_value = friedmanchisquare(df_matrix["combination_1"], df_matrix["combination_2"], df_matrix["combination_3"])

    results = [{"metric": metric_name, "test": "Friedman", "p_value": p_value}]
    print(f"\nFriedman Test for {metric_name}: p={p_value:.4f}")

    if p_value < 0.05:
        print(f"Significant difference found in {metric_name}, performing post-hoc tests...")

        # Nemenyi Test
        nemenyi_results = posthoc_nemenyi_friedman(df_matrix.values)
        print("\nNemenyi Post-Hoc Test Results:\n", nemenyi_results)

        for i in range(3):
            for j in range(i + 1, 3):
                results.append({"metric": metric_name, "test": "Nemenyi", "combination": f"combination_{i+1} vs combination_{j+1}", "p_value": nemenyi_results.iloc[i, j]})

    return results


# Main execution function
def main(projects, feature_combinations):
    classifiers = {
        'LR': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'RF': RandomForestClassifier(random_state=42),
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'GB': GradientBoostingClassifier(random_state=42)
    }

    all_results = []
    metric_names = ["accuracy", "precision", "recall", "f1_score", "mcc", "auroc"]

    # Store metrics across classifiers and projects
    combined_metrics = {metric: [] for metric in metric_names}

    for project in projects:
        train_file = f"{project}_training_file_entropy_bugs_bug_priorities_non_fatty_h1_final.csv"
        test_file = f"{project}_test_file_entropy_bugs_bug_priorities_non_fatty_h1_final.csv"
        print(f"\nProcessing project: {project}")

        data = preprocess_data(train_file, test_file, feature_combinations)

        for model_name, clf in classifiers.items():
            print(f"\nRunning {model_name} classifier on {project}...")
            metrics = train_and_predict(clf, data)

            # Store metrics for statistical tests
            for metric in metric_names:
                combined_metrics[metric].append([metrics[metric]["combination_1"], metrics[metric]["combination_2"], metrics[metric]["combination_3"]])

    # Convert combined metrics to matrices (40 × 3)
    for metric in metric_names:
        flattened_matrix = np.array(combined_metrics[metric]).reshape(40, 3)  # 40 rows, 3 columns
        all_results.extend(perform_statistical_tests(flattened_matrix, metric))

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("friedman_nemenyi_results_all_metrics_lr_svm_rf_xgb_gb_final.csv", index=False)
    print("\nAnalysis complete! Results saved.")


# Example usage
feature_combinations = [
    ['comm', 'adev', 'ddev', 'add', 'del', 'own', 'minor', 'ncomm', 'nadev', 'nddev', 'oexp', 'exp', "Change_entropy"],
    ['comm', 'adev', 'ddev', 'add', 'del', 'own', 'minor', 'ncomm', 'nadev', 'nddev', 'oexp', 'exp', "Co_change_entropy"],
    ['comm', 'adev', 'ddev', 'add', 'del', 'own', 'minor', 'ncomm', 'nadev', 'nddev', 'oexp', 'exp', "Change_entropy", "Co_change_entropy"]
]

projects = ["derby", "activemq", "pdfbox", "pig", "kafka", "maven", "struts", "nifi"]

main(projects, feature_combinations)
