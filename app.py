from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import pandas as pd
import os
import numpy as np

app = Flask(__name__)
app.secret_key = "co_prediction_secret_key"  # для работы с flash-сообщениями

# Загрузка модели и получение списка признаков
def load_model():
    model_path = os.path.join('models', 'co_prediction_model.pkl')
    features_path = os.path.join('models', 'model_features.txt')
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            feature_names = f.read().strip().split('\n')
        return model, feature_names
    else:
        return None, []

# Загрузка метрик модели
def load_metrics():
    metrics_path = os.path.join('models', 'model_metrics.txt')
    metrics = {}
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                metrics[key] = value
    
    return metrics

@app.route('/', methods=['GET', 'POST'])
def index():
    model, feature_names = load_model()
    prediction = None
    input_values = {}
    
    if model is None:
        flash("Ошибка: Модель не найдена. Убедитесь, что файлы модели существуют в папке 'models'.")
        return render_template('index.html', features=[])
    
    if request.method == 'POST':
        try:
            # Сбор данных из формы
            input_values = {}
            for feature in feature_names:
                value = request.form.get(feature, '')
                if not value:
                    flash(f"Ошибка: Поле '{feature}' не может быть пустым")
                    return render_template('index.html', features=feature_names)
                
                try:
                    input_values[feature] = float(value.replace(',', '.'))
                except ValueError:
                    flash(f"Ошибка: Значение '{value}' для поля '{feature}' должно быть числом")
                    return render_template('index.html', features=feature_names)
            
            # Создаем DataFrame из введенных данных
            input_df = pd.DataFrame([input_values])
            
            # Проверка на наличие всех необходимых признаков
            missing_features = set(feature_names) - set(input_df.columns)
            if missing_features:
                flash(f"Ошибка: Отсутствуют необходимые признаки: {', '.join(missing_features)}")
                return render_template('index.html', features=feature_names)
            
            # Прогнозирование
            prediction = model.predict(input_df)[0]
            
        except Exception as e:
            flash(f"Произошла ошибка при обработке запроса: {str(e)}")
    
    return render_template('index.html', features=feature_names, prediction=prediction, input_values=input_values)

@app.route('/stats')
def model_stats():
    # Проверяем наличие необходимых файлов
    required_files = [
        'correlation_matrix.png',
        'residuals_distribution.png',
        'prediction_scatter.png',
        'feature_importance.png'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join('static', f))]
    
    if missing_files:
        flash(f"Ошибка: Отсутствуют следующие файлы статистики: {', '.join(missing_files)}")
        return redirect(url_for('index'))
    
    # Получаем метрики модели
    metrics = load_metrics()
    
    return render_template('stats.html', metrics=metrics)

if __name__ == '__main__':
    # Создаем папки, если они не существуют
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True)