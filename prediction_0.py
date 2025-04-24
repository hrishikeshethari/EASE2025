import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

project_release = {
    "derby": ["10.3.1.4", "10.5.1.1"],
    "activemq": ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.5.0"],
    "pdfbox": ["1.5.0", "1.7.0", "1.8.0", "2.0.0"],
    "pig": ["release-0.6.0", "release-0.7.0", "release-0.8.0", "release-0.9.0"],
    "kafka": ["0.10.0.0", "0.11.0.0"],
    "maven": ["maven-3.1.0", "maven-3.3.9", "maven-3.5.0"],
    "struts": ["STRUTS_2_3_28", "STRUTS_2_3_32"],
    "nifi": ["nifi-0.5.0", "nifi-0.6.0", "nifi-0.7.0"]
}

for p in project_release:
    # Load the data
    train_file = f'{p}_training_file_entropy_bugs_bug_priorities_non_fatty_h1_final.csv'
    test_file = f'{p}_test_file_entropy_bugs_bug_priorities_non_fatty_h1_final.csv'
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Convert Bugs to binary
    train_df['Bugs'] = (train_df['Bugs'] > 0).astype(int)
    test_df['Bugs'] = (test_df['Bugs'] > 0).astype(int)

    # Separate features and target
    X_train = train_df.drop(columns=['File', 'Bugs'])
    y_train = train_df['Bugs']
    X_test = test_df.drop(columns=['File', 'Bugs'])
    y_test = test_df['Bugs']

    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns).fillna(0)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns).fillna(0)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # # Define feature combinations
    process_metrics = ['comm', 'adev', 'ddev', 'add', 'del', 'own', 'minor', 'ncomm', 'nadev',
                       'nddev', 'oexp', 'exp']
    feature_combinations = {
        'C4': process_metrics + ['Change_entropy', 'Co_change_entropy'],
        'C7': process_metrics + ['Change_entropy'],
        'C8': process_metrics + ['Co_change_entropy'],
    }

    # Define classifiers with parameter grids
    classifiers = {
        'LR': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'RF': RandomForestClassifier(random_state=42),
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'GB': GradientBoostingClassifier(random_state=42)
    }

    # Store the results
    results = []

    # Evaluate classifiers on each feature combination
    for clf_name, clf in classifiers.items():
        for combo_name, features in feature_combinations.items():
            X_train_combo = X_train_resampled[features]
            X_test_combo = X_test_scaled[features]

            best_clf = clf

            # Train and predict
            best_clf.fit(X_train_combo, y_train_resampled)
            y_pred = best_clf.predict(X_test_combo)
            y_prob = best_clf.predict_proba(X_test_combo)[:, 1] if hasattr(best_clf, 'predict_proba') else None

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auroc = roc_auc_score(y_test, y_prob)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Store the results
            results.append([clf_name, 'Accuracy', combo_name, f"{accuracy:.3f}"])
            results.append([clf_name, 'Precision', combo_name, f"{precision:.3f}"])
            results.append([clf_name, 'Recall', combo_name, f"{recall:.3f}"])
            results.append([clf_name, 'F1-Score', combo_name, f"{f1:.3f}"])
            results.append([clf_name, 'AUROC', combo_name, f"{auroc:.3f}"])
            results.append([clf_name, 'MCC', combo_name, f"{mcc:.3f}"])

    # Convert to DataFrame
    results_df = pd.DataFrame(results, columns=['Classifier', 'Metric', 'Combination', 'Score'])

    # Pivot the results for the required format
    pivot_df = results_df.pivot(index=['Metric', 'Classifier'], columns='Combination', values='Score').reset_index()

    # Save to CSV
    pivot_df.to_csv(f'{p}_prediction_results_entropy_non_fatty_smote_h1_final.csv', index=False)
    print(f"{p}_Results saved to prediction_results_final.csv.")
