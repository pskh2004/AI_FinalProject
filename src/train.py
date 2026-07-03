# src/train.py
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def get_models_and_grids(seed=42):
    """
    معرفی ۴ مدل مشخص شده در صورت‌پروژه به همراه فضای جستجوی هایپرپارامترها
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=seed, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=seed),
        "XGBoost": XGBClassifier(random_state=seed, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(random_state=seed, verbose=0)
    }

    grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10.0]
        },
        "Random Forest": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None]
        },
        "XGBoost": {
            "n_estimators": [50, 100],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1]
        },
        "CatBoost": {
            "iterations": [50, 100],
            "depth": [4, 6],
            "learning_rate": [0.05, 0.1]
        }
    }
    return models, grids


def train_and_tune_model(X_train, y_train, model_name, model, param_grid, seed=42):
    """
    آموزش مدل به کمک تکنیک اعتبارسنجی متقاطع (K-Fold CV) جهت بهینه‌سازی پارامترها و جلوگیری از بیش‌برازش
    """
    # تقسیم ۵ تایی داده‌ها (5-Fold CV) برای جلوگیری از Overfitting بر اساس صورت‌مسئله
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    # استفاده از معیار بهینه‌سازی f1 با توجه به ماهیت مسئله ریزش مشتری
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_