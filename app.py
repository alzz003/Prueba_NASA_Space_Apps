# app.py
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
import os
from utils import process_koi_lightcurve

app = Flask(__name__)
app.secret_key = 'super-secret-key-for-hackathon'
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Asegurarse de que la carpeta de subidas existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar el modelo entrenado
try:
    model = joblib.load('exoplanet_detector_model.joblib')
    print("Modelo cargado exitosamente.")
except FileNotFoundError:
    model = None
    print("ERROR: No se encontró 'exoplanet_detector_model.joblib'. Ejecuta el script de entrenamiento primero.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('El modelo no está disponible. Por favor, contacta al administrador.')
        return redirect(url_for('index'))

    if 'csv_file' not in request.files or request.files['csv_file'].filename == '':
        flash('No se seleccionó ningún archivo.')
        return redirect(url_for('index'))

    file = request.files['csv_file']
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df_new_koi = pd.read_csv(filepath)
            
            # Validar columnas necesarias
            required_cols = ['kepid', 'koi_period', 'koi_duration', 'koi_time0bk']
            if not all(col in df_new_koi.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df_new_koi.columns]
                flash(f'Error: Faltan las siguientes columnas en el CSV: {", ".join(missing)}')
                return redirect(url_for('index'))

            results = []
            for index, row in df_new_koi.iterrows():
                # Procesamiento en tiempo real
                flux_vector = process_koi_lightcurve(
                    kepid=row['kepid'],
                    period=row['koi_period'],
                    duration=row['koi_duration'],
                    transit_time=row['koi_time0bk']
                )
                
                prediction = "Error de Procesamiento"
                if flux_vector is not None:
                    # El modelo espera un array 2D
                    prediction_num = model.predict([flux_vector])[0]
                    prediction = "Planeta Potencial" if prediction_num == 1 else "Falso Positivo Probable"
                
                results.append({
                    'kepid': row['kepid'],
                    'period': row['koi_period'],
                    'prediction': prediction
                })

            return render_template('results.html', results=results, filename=filename)

        except Exception as e:
            flash(f'Ocurrió un error al procesar el archivo: {e}')
            return redirect(url_for('index'))

    else:
        flash('Formato de archivo no válido. Por favor, sube un archivo .csv')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)