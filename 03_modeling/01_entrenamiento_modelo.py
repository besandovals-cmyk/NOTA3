import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'dataset_maestro.parquet')
artifacts_dir = os.path.join(script_dir, '..', 'artifacts')

os.makedirs(artifacts_dir, exist_ok=True)

def main():
    print("⏳ Cargando Dataset Maestro...")
    df = pd.read_parquet(data_path, engine='pyarrow')
    print(f"✔ Datos cargados: {df.shape}")

    # ---------------------------------------------------------
    # 1. PREPROCESAMIENTO PROFESIONAL (LabelEncoder)
    # ---------------------------------------------------------
    # Guardaremos los diccionarios para que la API sepa traducir texto a números
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    encoders = {} 
    
    print(f"🔧 Entrenando codificadores para {len(cat_cols)} columnas...")
    
    for col in cat_cols:
        le = LabelEncoder()
        # Rellenamos nulos con 'MISSING' para que no fallen
        df[col] = df[col].fillna('MISSING').astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le 
        
    # Guardamos el diccionario de encoders
    encoders_path = os.path.join(artifacts_dir, 'encoders.joblib')
    joblib.dump(encoders, encoders_path)
    print(f"💾 Encoders guardados en: {encoders_path}")

    # ---------------------------------------------------------
    # 2. ENTRENAMIENTO
    # ---------------------------------------------------------
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']
    
    print("✂️ Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("🚀 Entrenando modelo LightGBM...")
    clf = lgb.LGBMClassifier(
        n_estimators=100,        
        learning_rate=0.05,      
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    clf.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 3. EVALUACIÓN Y GUARDADO
    # ---------------------------------------------------------
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n🏆 AUC-ROC Score Final: {auc:.4f}")
    
    # Guardar Modelo
    model_path = os.path.join(artifacts_dir, 'modelo_riesgo.joblib')
    joblib.dump(clf, model_path)
    
    # Guardar Columnas (Crucial para la API)
    cols_path = os.path.join(artifacts_dir, 'columnas_modelo.joblib')
    joblib.dump(X_train.columns.tolist(), cols_path)
    
    print("✅ Fase 2 completada.")

if __name__ == "__main__":
    main()