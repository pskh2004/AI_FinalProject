# src/evaluate.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(model, X_test, y_test):
    """
    محاسبه شاخص‌های ارزیابی بر روی داده‌های تست (که در بهینه‌سازی نقشی نداشته‌اند).
    """
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = y_pred

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_prob))
    }
    return metrics, y_pred