import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Define feature combinations and metrics
feature_combinations_all = [{"C7": "Change", "C8": "Co-change"}, {"C7": "Change", "C4": "P + C + Co"}]
for idx, feature_combinations in enumerate(feature_combinations_all):
    print(f"Index: {idx}, Value: {feature_combinations}")
    metrics = ["AUROC", "F1-Score", "MCC", "Precision", "Recall"]
    classifiers = ["LR", "SVM", "RF", "XGB", "GB"]
    projects = ["derby", "activemq", "pdfbox", "pig", "kafka", "maven", "struts", "nifi"]

    # Store results: {metric: {feature_combination: count}}
    final_results = {metric: defaultdict(int) for metric in metrics}

    # Store project names for labeling
    project_labels = {metric: defaultdict(list) for metric in metrics}

    # Process each project-classifier combination (40 total)
    for project in projects:
        file_path = f"{project}_prediction_results_entropy_non_fatty_smote_h1_final.csv"
        df = pd.read_csv(file_path)

        for metric in metrics:
            metric_df = df[df["Metric"] == metric]  # Filter relevant metric rows

            for clf in classifiers:
                clf_df = metric_df[metric_df["Classifier"] == clf]  # Get row for specific classifier

                if not clf_df.empty:
                    # Extract scores for C7, C8, and C4
                    scores = clf_df[list(feature_combinations.keys())].values.flatten()
                    best_fc = list(feature_combinations.keys())[scores.argmax()]  # Get best feature combination

                    # Increment count for the best feature combination
                    final_results[metric][feature_combinations[best_fc]] += 1

                    # Store project-classifier for labeling
                    project_labels[metric][feature_combinations[best_fc]].append(f"{project}-{clf}")

    # Create final combined pie charts
    if idx == 0:
        name = "c_co"
    else:
        name = "c_pcco"
    with PdfPages(f"{name}.pdf") as pdf:
        fig, axes = plt.subplots(1, 5, figsize=(25, 6))  # 5 pie charts in one page

        for ax, metric in zip(axes, metrics):
            sizes = [final_results[metric][fc] for fc in feature_combinations.values()]
            labels = [
                f"{fc} ({final_results[metric][fc]})\n"
                if final_results[metric][fc] > 0 else f"{fc} (0)"
                for fc in feature_combinations.values()
            ]

            ax.pie(sizes, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
            ax.set_title(metric)

        pdf.savefig()
        plt.close()
