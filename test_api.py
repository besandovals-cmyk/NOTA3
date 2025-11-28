import requests
import json

# URL de tu API local
url = "http://127.0.0.1:8000/evaluate_risk"

# Datos de un cliente ficticio (Solo ponemos algunos datos clave, el resto se rellenará con 0)
cliente_ejemplo = {
    "features": {
        "AMT_INCOME_TOTAL": 250000,
        "AMT_CREDIT": 1000000,
        "CODE_GENDER": "M",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "DAYS_EMPLOYED": -500,
        "EXT_SOURCE_2": 0.2,  # Un score externo bajo suele ser riesgoso
        "EXT_SOURCE_3": 0.1
    }
}

print("📡 Enviando solicitud a la API...")
try:
    response = requests.post(url, json=cliente_ejemplo)
    
    if response.status_code == 200:
        print("\n✅ ¡Respuesta Recibida!")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"\n❌ Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"❌ No se pudo conectar. ¿Está corriendo uvicorn? Error: {e}")