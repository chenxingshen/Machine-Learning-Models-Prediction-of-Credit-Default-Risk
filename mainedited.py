import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from lightgbm import early_stopping
import gc
from matplotlib.backends.backend_pdf import PdfPages


# ------------------ Data Loading & Preprocessing ------------------

def get_dataset(filename: str):
    data = pd.read_excel(filename, skiprows=1)
    data.rename(columns={"default payment next month": "DEFAULT"}, inplace=True)
    return data


def prepare_data(data, encoding="ohe"):
    features = data.drop(columns=["ID", "DEFAULT"])
    target = data["DEFAULT"]

    if encoding == "ohe":
        features = pd.get_dummies(features)
    elif encoding == "le":
        label_encoder = LabelEncoder()
        for col in features.columns:
            if features[col].dtype == "object":
                features[col] = label_encoder.fit_transform(features[col].astype(str))
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    return features, target


def split_data(features, target, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# ------------------ Model Training ------------------

def train_lightgbm(X_train, y_train, X_test, y_test, n_folds=5):
    feature_names = list(X_train.columns)
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(X_test.shape[0])
    valid_scores = []
    train_scores = []
    best_model = None
    best_valid_auc = float("-inf")

    for train_indices, valid_indices in k_fold.split(X_train):
        train_features, valid_features = X_train.iloc[train_indices], X_train.iloc[valid_indices]
        train_labels, valid_labels = y_train.iloc[train_indices], y_train.iloc[valid_indices]

        model = lgb.LGBMClassifier(
            n_estimators=10000,
            objective="binary",
            class_weight="balanced",
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            n_jobs=-1,
            random_state=50
        )

        model.fit(
            train_features, train_labels,
            eval_metric="auc",
            eval_set=[(train_features, train_labels), (valid_features, valid_labels)],
            eval_names=["train", "valid"],
            callbacks=[early_stopping(100, verbose=True)]
        )

        best_iteration = model.best_iteration_
        valid_auc = model.best_score_["valid"]["auc"]
        train_auc = model.evals_result_["train"]["auc"][best_iteration - 1]

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_model = model

        feature_importance_values += model.feature_importances_ / n_folds
        test_predictions += model.predict_proba(X_test, num_iteration=best_iteration)[:, 1] / n_folds
        valid_scores.append(valid_auc)
        train_scores.append(train_auc)

        gc.collect()

    metrics = pd.DataFrame({
        "fold": list(range(n_folds)),
        "train": train_scores,
        "valid": valid_scores,
    })
    metrics.loc["overall"] = ["overall", metrics["train"].mean(), metrics["valid"].mean()]

    feature_importances = pd.DataFrame({"feature": feature_names, "importance": feature_importance_values})
    return best_model, feature_importances, metrics


def train_svm(X_train, y_train):
    svm_model = SVC(kernel="rbf", probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model


def train_logistic_regression(X_train, y_train):
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="roc_auc")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def train_hybrid_model(X_train, y_train):
    tree_model = ExtraTreesClassifier(n_estimators=50, max_depth=5, random_state=42)
    tree_model.fit(X_train, y_train)
    tree_probs = tree_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
    logreg_model = LogisticRegression(random_state=42)
    logreg_model.fit(tree_probs, y_train)

    return tree_model, logreg_model


# ------------------ Evaluation & Visualization ------------------

def evaluate_model(model, X_test, y_test, model_name, use_proba=False):
    if use_proba:
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities
        y_pred = (y_pred_proba > 0.5).astype(int)  # Convert to binary
    else:
        y_pred = model.predict(X_test) # Direct binary classification

    report = classification_report(y_test, y_pred)
    print(f"\nClassification Report - {model_name}:\n{report}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    return y_pred


def plot_roc_curve(model, X_test, y_test, model_name):
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.show()

    return roc_auc


def plot_feature_importances(feature_importances):
    feature_importances = feature_importances.sort_values("importance", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances["feature"][:15], feature_importances["importance"][:15], color="blue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances - LightGBM")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.show()


# ------------------ Main Execution ------------------

if __name__ == "__main__":
    # Load dataset
    data = get_dataset("data.xls")

    # Prepare data
    features, target = prepare_data(data, encoding="ohe")

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(features, target)

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    model_performance = [] # store metrics for comparison

    # Train and evaluate LightGBM model
    print("Training LightGBM model...")
    best_lgbm_model, feature_importances, lgbm_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
    y_pred_lgbm = evaluate_model(best_lgbm_model, X_test, y_test,"LightGBM", use_proba=True) # Ensure fair comparison
    auc_lgbm = plot_roc_curve(best_lgbm_model, X_test, y_test, "LightGBM")
    plot_feature_importances(feature_importances)

    model_performance.append(("LightGBM", auc_lgbm, classification_report(y_test, y_pred_lgbm, output_dict=True)))

    # Train and evaluate SVM (Support Vector Machine) model with RBF kernel
    print("Training SVM model...")
    svm_model = train_svm(X_train_scaled, y_train)
    y_pred_svm = evaluate_model(svm_model, X_test_scaled, y_test, "SVM")
    auc_svm = plot_roc_curve(svm_model, X_test_scaled, y_test, "SVM")

    model_performance.append(("SVM", auc_svm, classification_report(y_test, y_pred_svm, output_dict=True)))

    # Train and evaluate Non-Linear Logistic Regression model
    print("Training Non-Linear Logistic Regression model...")
    logreg_model = train_logistic_regression(X_train_scaled, y_train)
    y_pred_logreg = evaluate_model(logreg_model, X_test_scaled, y_test, "Logistic Regression")
    auc_logreg = plot_roc_curve(logreg_model, X_test_scaled, y_test, "Logistic Regression")

    model_performance.append(("Logistic Regression", auc_logreg, classification_report(y_test, y_pred_logreg, output_dict=True)))

    # Train and evaluate Hybrid Model (Decision Tree + Logistic Regression)
    print("Training Hybrid Model (Decision Tree + Logistic Regression)...")
    tree_model, hybrid_logreg = train_hybrid_model(X_train, y_train)
    tree_probs = tree_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
    y_pred_hybrid = evaluate_model(hybrid_logreg, tree_probs, y_test, "Hybrid Model")
    auc_hybrid = plot_roc_curve(hybrid_logreg, tree_probs, y_test, "Hybrid Model")

    model_performance.append(("Hybrid Model", auc_hybrid, classification_report(y_test, y_pred_hybrid, output_dict=True)))


    exit(0)
    # Create PDF report
    pdf_filename = "Credit_Scoring_Analysis.pdf"
    with PdfPages(pdf_filename) as pdf_pages:
        # Find model with highest AUC
        best_model = max(model_performance, key=lambda x: x[1])

        summary_text = (
        "Credit Scoring Model Comparison\n\n"
        f"Best Model: {best_model[0]:<22} Highest AUC Score: {best_model[1]:.4f}\n\n"
        "Overall Performance Table:\n"
        "----------------------------------------------------------------------\n"
        "| Model                  | AUC Score | Precision | Recall | F1-Score |\n"
        "----------------------------------------------------------------------\n"
        )

        for name, auc, metrics in model_performance:
            precision = metrics["1"]["precision"]
            recall = metrics["1"]["recall"]
            f1_score = metrics["1"]["f1-score"]
            summary_text += f"| {name:<22} | {auc:.4f} | {precision:.4f} | {recall:.4f} | {f1_score:.4f} |\n"
        summary_text += "----------------------------------------------------------------------"

        fig_summary = plt.figure(figsize=(10, 6))
        plt.axis("off")
        plt.text(0.05, 0.5, summary_text, fontsize=12, verticalalignment="center", horizontalalignment="left", fontfamily="monospace", wrap=True)
        pdf_pages.savefig(fig_summary)
        plt.close(fig_summary)

    print(f"PDF report saved as {pdf_filename}")
