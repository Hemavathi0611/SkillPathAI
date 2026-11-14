import json
import pandas as pd

from pathlib import Path

REPORT_DIR = Path("reports/careerpath")


def load_csv(filename):
    path = REPORT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")

    return pd.read_csv(path)


def main():

    print("BUILDING CAREERPATH EVALUATION SUMMARY")

    # 1. Baseline metrics
    with open(
        REPORT_DIR / "metrics.json",
        encoding="utf-8"
    ) as file:
        baseline = json.load(file)

    # 2. Load experiment reports
    ablation = load_csv("feature_ablation_results.csv")
    individual = load_csv("individual_feature_results.csv")
    novelty = load_csv("feature_novelty_results.csv")
    sanity = load_csv("sanity_check_results.csv")
    cross_validation = load_csv("cross_validation_results.csv")

    # 3. Calculate cross-validation statistics
    cv_summary = {}

    for metric in [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_F1"
    ]:
        values = cross_validation[metric]

        cv_summary[metric] = {
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std(ddof=0)), 6)
        }

    # 4. Prepare consolidated experiment table
    summary_rows = []

    summary_rows.append({
        "Experiment": "Baseline Logistic Regression",
        "Records": 1800,
        "Accuracy": baseline["accuracy"],
        "Balanced_Accuracy": baseline["balanced_accuracy"],
        "Macro_F1": baseline["macro_f1"],
        "Notes": "Original 80:20 stratified test split"
    })

    for _, row in ablation.iterrows():

        summary_rows.append({
            "Experiment": f"Feature Ablation: {row['Experiment']}",
            "Records": 1800,
            "Accuracy": row["Accuracy"],
            "Balanced_Accuracy": row["Balanced_Accuracy"],
            "Macro_F1": row["Macro_F1"],
            "Notes": f"Features used: {row['Feature_Count']}"
        })

    for _, row in sanity.iterrows():

        summary_rows.append({
            "Experiment": f"Sanity Check: {row['Experiment']}",
            "Records": 1800,
            "Accuracy": row["Accuracy"],
            "Balanced_Accuracy": row["Balanced_Accuracy"],
            "Macro_F1": row["Macro_F1"],
            "Notes": "Comparison against original-label model"
        })

    for _, row in novelty.iterrows():

        summary_rows.append({
            "Experiment": f"Feature Novelty: {row['Group']}",
            "Records": row["Unseen_Records"],
            "Accuracy": row["Accuracy"],
            "Balanced_Accuracy": row["Balanced_Accuracy"],
            "Macro_F1": row["Macro_F1"],
            "Notes": "Evaluation on unseen feature values/combinations"
        })

    summary_rows.append({
        "Experiment": "5-Fold Cross-Validation Mean",
        "Records": 9000,
        "Accuracy": cv_summary["Accuracy"]["mean"],
        "Balanced_Accuracy": cv_summary["Balanced_Accuracy"]["mean"],
        "Macro_F1": cv_summary["Macro_F1"]["mean"],
        "Notes": "Standard deviation: " +
                 str(cv_summary["Accuracy"]["std"])
    })

    consolidated = pd.DataFrame(summary_rows)

    # 5. Save consolidated experiment results
    csv_output = REPORT_DIR / "consolidated_evaluation.csv"
    consolidated.to_csv(csv_output, index=False)

    # 6. Save detailed JSON summary
    json_summary = {
        "project": "SkillPath AI - CareerPath",
        "model": "Logistic Regression",
        "dataset_records": 9000,
        "career_classes": 30,
        "baseline_test_metrics": baseline,
        "cross_validation": cv_summary,
        "individual_feature_experiments": individual.to_dict(
            orient="records"
        ),
        "feature_ablation_experiments": ablation.to_dict(
            orient="records"
        ),
        "feature_novelty_experiments": novelty.to_dict(
            orient="records"
        ),
        "sanity_check_experiments": sanity.to_dict(
            orient="records"
        ),
        "evaluation_limitation": (
            "All experiments use the same supplied dataset. "
            "High scores do not establish real-world career "
            "recommendation validity. Independent validation "
            "with real student outcomes is required."
        )
    }

    json_output = REPORT_DIR / "evaluation_summary.json"

    with open(
        json_output,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            json_summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # 7. Display results
    print("\nCONSOLIDATED EVALUATION")
    print(
        consolidated[
            [
                "Experiment",
                "Records",
                "Accuracy",
                "Macro_F1"
            ]
        ].round(4).to_string(index=False)
    )

    print("\nCROSS-VALIDATION SUMMARY")

    for metric, values in cv_summary.items():
        print(
            f"{metric}: "
            f"{values['mean']:.4f} "
            f"+/- {values['std']:.4f}"
        )

    print("\nSAVED FILES")
    print(csv_output)
    print(json_output)

    print("\nEVALUATION SUMMARY COMPLETED")


if __name__ == "__main__":
    main()
