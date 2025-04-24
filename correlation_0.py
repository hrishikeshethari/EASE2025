import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

# Define project releases
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

# List to store correlation results
all_results = []

# Loop through each project
for project in project_release:
    # Load CSV file
    try:
        df_bugs = pd.read_csv(f"{project}_train_test_file_entropy_bugs_bug_priorities_non_fatty_h1_final.csv")
    except FileNotFoundError:
        print(f"Files for project {project} not found. Skipping...")
        continue

    # Check if required columns exist
    required_columns = ["Change_entropy", "Co_change_entropy", "Bugs"]
    if not all(col in df_bugs.columns for col in required_columns):
        print(f"Missing columns in {project}. Skipping...")
        continue

    X = df_bugs[['Change_entropy', 'Co_change_entropy']]  # Features
    y = df_bugs['Bugs']  # Target (Binary)

    # Compute Correlations
    valid_change = df_bugs["Change_entropy"].notnull() & df_bugs["Bugs"].notnull()
    valid_cochange = df_bugs["Co_change_entropy"].notnull() & df_bugs["Bugs"].notnull()

    def compute_correlation(method, valid_mask, col):
        if valid_mask.sum() >= 2:
            return method(df_bugs.loc[valid_mask, col], df_bugs.loc[valid_mask, "Bugs"])
        else:
            return float('nan'), float('nan')

    # Compute all correlations
    pearson_change, p_pearson_change = compute_correlation(pearsonr, valid_change, "Change_entropy")
    spearman_change, p_spearman_change = compute_correlation(spearmanr, valid_change, "Change_entropy")
    kendall_change, p_kendall_change = compute_correlation(kendalltau, valid_change, "Change_entropy")

    pearson_cochange, p_pearson_cochange = compute_correlation(pearsonr, valid_cochange, "Co_change_entropy")
    spearman_cochange, p_spearman_cochange = compute_correlation(spearmanr, valid_cochange, "Co_change_entropy")
    kendall_cochange, p_kendall_cochange = compute_correlation(kendalltau, valid_cochange, "Co_change_entropy")

    print(f"{project} | Pearson: ({pearson_change:.4f}, {pearson_cochange:.4f}) | p-values: ({p_pearson_change:.4e}, {p_pearson_cochange:.4e})")
    print(f"{project} | Spearman: ({spearman_change:.4f}, {spearman_cochange:.4f}) | p-values: ({p_spearman_change:.4e}, {p_spearman_cochange:.4e})")
    print(f"{project} | Kendall Tau: ({kendall_change:.4f}, {kendall_cochange:.4f}) | p-values: ({p_kendall_change:.4e}, {p_kendall_cochange:.4e})")

    # Store results
    all_results.append([project, "Pearson", f"{pearson_change:.3f}", f"{p_pearson_change:.3e}", f"{pearson_cochange:.3f}",
                        f"{p_pearson_cochange:.3e}"])
    all_results.append([project, "Spearman", f"{spearman_change:.3f}", f"{p_spearman_change:.3e}", f"{spearman_cochange:.3f}",
                        f"{p_spearman_cochange:.3e}"])
    all_results.append([project, "Kendall Tau", f"{kendall_change:.3f}", f"{p_kendall_change:.3e}", f"{kendall_cochange:.3f}",
                        f"{p_kendall_cochange:.3e}"])

# Create DataFrame & Save to CSV
columns = ["Project", "Correlation_Type", "Change_Entropy", "Change_p_value", "Co_Change_Entropy", "Co_Change_p_value"]
df_results = pd.DataFrame(all_results, columns=columns)
df_results.to_csv("entropy_correlation_results_hyper_entropy_smote_non_fatty_final_count.csv", index=False)

print("Correlation results saved to entropy_correlation_results_hyper_entropy_smote_non_fatty_final.csv")
