import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import joblib
import os

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'dataset_maestro.parquet')
artifacts_dir = os.path.join(script_dir, '..', 'artifacts')
output_images_dir = os.path.join(script_dir, 'reportes_visuales')

# Crear carpeta para guardar los gráficos
os.makedirs(output_images_dir, exist_ok=True)

def main():
    print("🔍 INICIANDO EVALUACIÓN INDEPENDIENTE")
    
    # 1. CARGAR DATOS Y ARTEFACTOS
    print("⏳ Cargando datos y modelo entrenado...")
    df = pd.read_parquet(data_path, engine='pyarrow')
    
    try:
        model = joblib.load(os.path.join(artifacts_dir, 'modelo_riesgo.joblib'))
        encoders = joblib.load(os.path.join(artifacts_dir, 'encoders.joblib'))
    except FileNotFoundError:
        print("❌ Error: No se encuentran los artefactos. Ejecuta la Fase 4 primero.")
        return

    # 2. REPLICAR PREPROCESAMIENTO (Usando los encoders guardados)
    print("⚙️ Aplicando transformaciones...")
    for col, le in encoders.items():
        if col in df.columns:
            # Convertimos a string, rellenamos nulos y mapeamos
            df[col] = df[col].fillna('MISSING').astype(str)
            
            # Usamos un mapeo seguro para evitar errores si hay datos nuevos (aunque aquí es el mismo dataset)
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # 3. SPLIT (Debe ser idéntico al del entrenamiento: random_state=42)
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. GENERAR PREDICCIONES
    print("🚀 Generando predicciones de prueba...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 5. GRAFICAR MATRIZ DE CONFUSIÓN
    print("📊 Generando Matriz de Confusión...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión (Test Set)')
    plt.xlabel('Predicción (0=Paga, 1=Default)')
    plt.ylabel('Realidad')
    
    cm_path = os.path.join(output_images_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"   ✔ Guardada en: {cm_path}")
    plt.close()

    # 6. GRAFICAR CURVA ROC
    print("📈 Generando Curva ROC...")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - Riesgo de Crédito')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(output_images_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    print(f"   ✔ Guardada en: {roc_path}")
    plt.close()
    
    # 7. REPORTE DE TEXTO
    report_path = os.path.join(output_images_dir, 'metricas_finales.txt')
    with open(report_path, 'w') as f:
        f.write("REPORTE DE EVALUACIÓN DEL MODELO\n")
        f.write("================================\n\n")
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nAUC-ROC Score: {roc_auc:.4f}")
    
    print(f"📝 Reporte de texto guardado en: {report_path}")
    print("✅ Fase de Evaluación Completada.")

if __name__ == "__main__":
    main()