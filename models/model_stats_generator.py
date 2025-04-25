import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

def generate_model_stats(dataset_path='AirQuality.csv'):
    """Генерирует статистику и графики для модели и сохраняет их в соответствующие папки"""
    
    # Создание директорий для сохранения результатов
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Загрузка данных
    df = pd.read_csv(dataset_path, sep=';')
    
    # Предобработка данных (как в исходном коде)
    if 'Unnamed: 15' in df.columns and 'Unnamed: 16' in df.columns:
        df = df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1)
    
    df['CO(GT)'] = pd.to_numeric(df['CO(GT)'].str.replace(',', '.'), errors='coerce')
    df['C6H6(GT)'] = pd.to_numeric(df['C6H6(GT)'].str.replace(',', '.'), errors='coerce')
    df['T'] = pd.to_numeric(df['T'].str.replace(',', '.'), errors='coerce')
    df['RH'] = pd.to_numeric(df['RH'].str.replace(',', '.'), errors='coerce')
    df['AH'] = pd.to_numeric(df['AH'].str.replace(',', '.'), errors='coerce')
    
    df.replace(-200, np.nan, inplace=True)
    df = df.dropna()
    
    # Сохранение описательной статистики
    stats = df.describe()
    stats.to_csv('models/descriptive_stats.csv')
    
    # Создание корреляционной матрицы
    corr_matrix = df.drop(columns=['Date', 'Time']).corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Корреляционная матрица")
    plt.tight_layout()
    plt.savefig('static/correlation_matrix.png')
    plt.close()
    
    # Подготовка данных для обучения модели
    X = df.drop(columns=['CO(GT)', 'Date', 'Time'])
    y = df['CO(GT)']
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Сохранение модели и списка признаков
    joblib.dump(model, 'models/co_prediction_model.pkl')
    with open('models/model_features.txt', 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    
    # Предсказания на тестовой выборке
    y_pred = model.predict(X_test)
    
    # Расчет метрик качества
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Сохранение метрик в файл
    metrics = {
        'MAPE': f"{mape:.2f}%",
        'RMSE': f"{rmse:.4f}",
        'MAE': f"{mae:.4f}",
        'R^2': f"{r2:.4f}"
    }
    with open('models/model_metrics.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    # Визуализация распределения остатков
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='red')
    plt.title('Распределение остатков')
    plt.xlabel('Остатки')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.savefig('static/residuals_distribution.png')
    plt.close()
    
    # График рассеяния: реальные vs предсказанные значения
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Реальные значения CO')
    plt.ylabel('Предсказанные значения CO')
    plt.title('Сравнение реальных и предсказанных значений')
    plt.grid(True)
    plt.savefig('static/prediction_scatter.png')
    plt.close()
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'Признак': X.columns,
        'Важность': np.abs(model.coef_)
    }).sort_values(by='Важность', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Важность', y='Признак', data=feature_importance)
    plt.title('Важность признаков')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.close()
    
    print("Статистика модели успешно сгенерирована и сохранена")
    return metrics

if __name__ == "__main__":
    # Если скрипт запущен напрямую, запрашиваем путь к файлу данных
    file_path = input("Введите путь к файлу данных (AirQuality.csv): ")
    if not file_path:
        file_path = 'AirQuality.csv'
    
    if os.path.exists(file_path):
        generate_model_stats(file_path)
    else:
        print(f"Ошибка: файл {file_path} не найден")