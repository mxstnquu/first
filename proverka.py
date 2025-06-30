import pandas as pd
import joblib
from catboost import Pool, CatBoostClassifier


def test_saved_model():
    try:
        # Загрузка модели и метаданных
        loaded_data = joblib.load('churn_model.pkl')
        model = loaded_data['model']
        cat_cols = loaded_data['cat_cols']

        # Загрузка и подготовка данных
        df = pd.read_csv('input_123.csv')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)

        # Проверка на пустые данные
        if df.empty:
            raise ValueError("Данные пусты после обработки")

        # Подготовка признаков
        X = df.drop(['customerID', 'Churn'], axis=1)
        y = df['Churn'].map({'Yes': 1, 'No': 0})

        # Используем ПЕРВУЮ строк
        sample = X.iloc[[0]]
        true_label = y.iloc[0]

        #специальный пул данных
        sample_pool = Pool(sample, cat_features=cat_cols)
        pred = model.predict(sample_pool)
        proba = model.predict_proba(sample_pool)

        print(f"\nМодель успешно загружена. Тестовое предсказание:")
        print(f"   Предсказанный класс: {pred[0]}")
        print(f"   Вероятность положительного класса: {proba[0][1]:.4f}")
        print(f"   Истинный класс: {true_label}")
        return True
    except Exception as e:
        print(f"\nОшибка при тестировании модели: {e}")
        return False


# Запуск теста
if __name__ == "__main__":
    if test_saved_model():
        print("Модель работает корректно.")
    else:
        print("Обнаружены ошибки в работе модели.")