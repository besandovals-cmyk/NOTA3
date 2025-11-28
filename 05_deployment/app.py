from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# 1. CONFIGURACIÓN E INICIALIZACIÓN
app = FastAPI(
    title="API de Riesgo Crediticio",
    description="Microservicio para predecir Default usando LightGBM",
    version="1.0.0"
)

# Definimos rutas relativas para buscar los artefactos
script_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(script_dir, '..', 'artifacts')

print("⏳ Cargando modelo y herramientas...")
try:
    # Cargamos el modelo, los traductores de texto (encoders) y la lista de columnas
    model = joblib.load(os.path.join(artifacts_dir, 'modelo_riesgo.joblib'))
    encoders = joblib.load(os.path.join(artifacts_dir, 'encoders.joblib'))
    model_columns = joblib.load(os.path.join(artifacts_dir, 'columnas_modelo.joblib'))
    print("✅ API lista y artefactos cargados.")
except Exception as e:
    print(f"❌ Error fatal cargando artefactos: {e}")
    # Esto hará que la API falle al inicio si no tiene los archivos, lo cual es bueno para debug
    raise e

# 2. DEFINIR EL FORMATO DE LOS DATOS DE ENTRADA
# Usamos un diccionario flexible para no tener que escribir las 200 columnas a mano aquí
class ClientData(BaseModel):
    features: dict 

# 3. ENDPOINT DE PREDICCIÓN
@app.post("/evaluate_risk")
def predict(data: ClientData):
    try:
        # A. Convertir el JSON recibido en un DataFrame de Pandas
        input_df = pd.DataFrame([data.features])
        
        # B. Preprocesamiento (CRÍTICO: Debe ser idéntico al entrenamiento)
        
        # 1. Alinear columnas: 
        # El modelo espera exactamente las mismas columnas en el mismo orden.
        # Si el JSON no trae alguna columna, la rellenamos con 0.
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # 2. Aplicar los Encoders (Texto -> Número)
        for col, le in encoders.items():
            if col in input_df.columns:
                # Convertimos a string y manejamos nulos
                input_df[col] = input_df[col].fillna('MISSING').astype(str)
                
                # Truco de seguridad: Usamos un diccionario de mapeo.
                # Si llega un valor desconocido (ej: "Gender: Alien"), se convierte en 0 para no romper la API.
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                input_df[col] = input_df[col].map(mapping).fillna(0).astype(int)

        # C. Predicción
        # predict_proba devuelve [[prob_clase_0, prob_clase_1]]
        probabilidad = float(model.predict_proba(input_df)[:, 1][0])
        
        # Umbral de decisión (puedes ajustarlo, 0.50 es estándar)
        decision = "RECHAZAR" if probabilidad > 0.50 else "APROBAR"
        
        return {
            "decision": decision,
            "probabilidad_default": round(probabilidad, 4),
            "riesgo": "ALTO" if probabilidad > 0.50 else "BAJO",
            "mensaje": "Evaluación completada exitosamente"
        }

    except Exception as e:
        # Si algo falla, devolvemos un error 500 con el detalle
        raise HTTPException(status_code=500, detail=f"Error procesando solicitud: {str(e)}")

# 4. RUTA DE PRUEBA (Health Check)
@app.get("/")
def health_check():
    return {"status": "ok", "service": "Credit Risk API running"}