import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# Загрузка данных
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Обработка TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Создание метаданных
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

synthesizer = GaussianCopulaSynthesizer(metadata)

# Обучение модели
synthesizer.fit(df)

# Генерация синтетических данных
synthetic_data = synthesizer.sample(num_rows=140000)

# Сохранение
synthetic_data.to_csv('synthetic_telco_churn_140k.csv', index=False)
