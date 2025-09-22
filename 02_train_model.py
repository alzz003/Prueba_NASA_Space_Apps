# 02_train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    """
    Carga el dataset procesado y entrena un modelo XGBoost para detectar exoplanetas.
    """
    print("Cargando el dataset de curvas de luz procesadas...")
    df = pd.read_csv('./data/processed_lightcurves.csv')
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print("Entrenando el modelo XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Entrenamiento completado. Evaluando rendimiento...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No Planeta', 'Planeta'])
    
    print(f"\nPrecisión del Modelo: {accuracy * 100:.2f}%")
    print("\nReporte de Clasificación:")
    print(report)
    
    print("Guardando el modelo entrenado...")
    joblib.dump(model, 'exoplanet_detector_model.joblib')
    
    print("\n¡Modelo guardado exitosamente como 'exoplanet_detector_model.joblib'!")

if __name__ == '__main__':
    train_model()