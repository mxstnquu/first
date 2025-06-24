import pandas as pd
import matplotlib.pyplot as plt
import shap
import time
import joblib
import warnings
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             ConfusionMatrixDisplay, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from catboost import CatBoostClassifier, Pool

# Отключение предупреждений
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    #Загрузка и предварительная обработка данных.
    df = pd.read_csv(filepath)

    # Преобразование TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    cat_cols = df.select_dtypes(['object']).columns.difference(['Churn', 'customerID'])

    # Создаем две версии данных:
    #Для большинства моделей
    df_num = df.copy()
    for col in cat_cols:
        df_num[col] = df_num[col].astype('category').cat.codes

    # Для CatBoost
    df_str = df.copy()

    # Подготовка признаков и целевой переменной
    X_num = df_num.drop(['customerID', 'Churn'], axis=1)
    X_str = df_str.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    return X_num, X_str, y, cat_cols.tolist()


def train_and_evaluate_models(models, model_data, y_train, y_test):
    #Обучение и оценка моделей
    metrics = []
    trained_models = {}

    for name, model in models.items():
        start_time = time.time()
        X_train, X_test = model_data[name]

        # Обучение модели
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Предсказания
        y_pred = model.predict(X_test)

        # Расчет вероятностей
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_roc = roc_auc_score(y_test, y_proba)
        else:
            auc_roc = None

        # Сохранение метрик
        metrics.append({
            "Model": name,
            "Train Time (s)": round(time.time() - start_time, 3),
            "AUC-ROC": auc_roc,
            "F1-score": f1_score(y_test, y_pred),
            "Accuracy": accuracy_score(y_test, y_pred)
        })

    return pd.DataFrame(metrics), trained_models


def calibrate_models(models, model_data, y_train, y_test):
    #Калибровка моделей и оценка калиброванных версий.
    metrics = []
    calibration_curves = {}
    calibrated_models = {}

    for name, model in models.items():
        start_time = time.time()
        X_train, X_test = model_data[name]

        # Калибровка (кроме логистической регрессии)
        if name == 'Logistic Regression':
            calibrated_model = model
        else:
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                method='sigmoid',
                cv=5
            )
            calibrated_model.fit(X_train, y_train)

        calibrated_models[name] = calibrated_model

        # Предсказания
        y_pred = calibrated_model.predict(X_test)
        y_proba = calibrated_model.predict_proba(X_test)[:, 1]

        # Расчет метрик
        auc_roc = roc_auc_score(y_test, y_proba)
        brier_score = ((y_proba - y_test) ** 2).mean()

        # Калибровочная кривая
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        calibration_curves[name] = (prob_true, prob_pred)

        # Сохранение метрик
        metrics.append({
            "Model": name,
            "Train Time (s)": round(time.time() - start_time, 3),
            "AUC-ROC": auc_roc,
            "F1-score": f1_score(y_test, y_pred),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Brier Score": brier_score
        })

    return pd.DataFrame(metrics), calibration_curves, calibrated_models


def plot_calibration_curves(calibration_curves):
    #Визуализация калибровочных кривых
    plt.figure(figsize=(10, 6))
    for name, (prob_true, prob_pred) in calibration_curves.items():
        plt.plot(prob_pred, prob_true, marker='o', label=name)

    plt.plot([0, 1], [0, 1], linestyle='--', label='Идеальная калибровка')
    plt.title('Калибровочные кривые')
    plt.xlabel('Средняя предсказанная вероятность')
    plt.ylabel('Доля положительных классов')
    plt.legend()
    plt.grid()
    plt.show()


# Основной блок выполнения
if __name__ == "__main__":
    # Загрузка и подготовка данных
    X_num, X_str, y, cat_cols = load_and_preprocess_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Разделение данных
    # Для большинства моделей
    X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_num, y, test_size=0.25, stratify=y, random_state=42
    )
    # Для CatBoost
    X_train_str, X_test_str, _, _ = train_test_split(
        X_str, y, test_size=0.25, stratify=y, random_state=42
    )

    # Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    #Подготовка данных для каждой модели
    model_data = {
        'XGBoost': (X_train_scaled, X_test_scaled),
        'Logistic Regression': (X_train_scaled, X_test_scaled),
        'Random Forest': (X_train_num, X_test_num),
        'KNeighbors': (X_train_scaled, X_test_scaled),
        'Decision Tree': (X_train_num, X_test_num),
        'Naive Bayes': (X_train_num, X_test_num),
        'CatBoost': (X_train_str, X_test_str)
    }

    # Инициализация моделей
    models = {
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight='balanced_subsample',
            random_state=42
        ),
        'KNeighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        ),
        'Naive Bayes': GaussianNB(),
        'CatBoost': CatBoostClassifier(
            cat_features=cat_cols,
            random_seed=42,
            verbose=0,
            auto_class_weights='Balanced',
            iterations=1000,
            early_stopping_rounds=50
        )
    }

    #Обучение и оценка моделей
    base_metrics, base_models = train_and_evaluate_models(
        models,
        model_data,
        y_train,
        y_test
    )

    #Калибровка моделей
    cal_metrics, cal_curves, cal_models = calibrate_models(
        base_models,
        model_data,
        y_train,
        y_test
    )

    #Визуализация результатов
    # Распределение классов
    plt.figure(figsize=(6, 4))
    y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Распределение классов')
    plt.xticks([0, 1], ['Не ушли (0)', 'Ушли (1)'], rotation=0)
    plt.show()

    # Матрицы ошибок
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    models_to_plot = ['XGBoost', 'Logistic Regression', 'Random Forest', 'Decision Tree']

    for ax, name in zip(axes.flatten(), models_to_plot):
        model = cal_models[name]
        X_train, X_test = model_data[name]

        ConfusionMatrixDisplay.from_estimator(
            estimator=model,
            X=X_test,
            y=y_test,
            ax=ax,
            colorbar=False
        )
        ax.set_title(name)

    plt.tight_layout()
    plt.show()

    # Калибровочные кривые
    plot_calibration_curves(cal_curves)

    # Сравнение точности
    accuracy_scores = {}
    for name, model in cal_models.items():
        _, X_test = model_data[name]
        accuracy_scores[name] = accuracy_score(y_test, model.predict(X_test))

    plt.figure(figsize=(10, 5))
    plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color='lightgreen')
    plt.title('Сравнение точности моделей')
    plt.ylabel('Accuracy')
    plt.ylim(0.6, 0.9)
    plt.xticks(rotation=45)
    plt.show()

    # SHAP-анализ для XGBoost
    explainer = shap.TreeExplainer(base_models['XGBoost'])
    shap_values = explainer.shap_values(X_test_scaled)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=X_num.columns,
        plot_type="bar",
        show=False
    )
    plt.title("XGBoost - Важность признаков")
    plt.tight_layout()
    plt.show()

    # Вывод результатов
    print("\nРезультаты сравнения моделей (до калибровки):")
    print(base_metrics.sort_values(by="AUC-ROC", ascending=False))
    print("\nРезультаты сравнения моделей (после калибровки):")
    print(cal_metrics.sort_values(by="AUC-ROC", ascending=False))

    # Сохранение лучшей модел
    best_model = cal_models['CatBoost']
    metadata = {
        'model': best_model,
        'scaler': scaler,
        'cat_cols': cat_cols,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(metadata, 'churn_model.pkl')  # Всё в одном файлеё


    # Проверка сохраненной модели
