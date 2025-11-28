import os

# Estructura de carpetas basada estrictamente en los requisitos del examen [cite: 13-19]
structure = [
    "01_data_understanding", # Scripts para EDA
    "02_data_preparation",   # Scripts de limpieza e ingeniería de características
    "03_modeling",           # Scripts para entrenamiento y validación
    "04_evaluation",         # Evaluación final y reportes
    "05_deployment",         # Código de la API (app.py)
    "artifacts",             # Modelos entrenados (.pkl), scalers, etc.
    "data"                   # (Opcional pero recomendado) Para guardar los csv localmente sin subirlos
]

# Archivos base requeridos [cite: 20, 21]
files = {
    "README.md": "# Proyecto: Predicción de Riesgo de Incumplimiento de Crédito\n\nEstructura basada en CRISP-DM para el examen de Machine Learning.",
    "requirements.txt": "pandas\nnumpy\nscikit-learn\nmatplotlib\nseaborn\nflask\nfastapi\nuvicorn\nlightgbm\nxgboost\nshap",
    ".gitignore": "venv/\n__pycache__/\n.DS_Store\n*.csv\n/data/\n*.pkl\n*.joblib\n.ipynb_checkpoints/"
}

def create_structure():
    # 1. Crear directorios
    for folder in structure:
        os.makedirs(folder, exist_ok=True)
        # Crear un archivo .gitkeep para que Git reconozca la carpeta vacía
        with open(os.path.join(folder, ".gitkeep"), "w") as f:
            pass
        print(f"✔ Carpeta creada: /{folder}")

    # 2. Crear archivos base
    for filename, content in files.items():
        with open(filename, "w") as f:
            f.write(content)
        print(f"✔ Archivo creado: {filename}")

    print("\n--- Estructura del proyecto inicializada con éxito ---")

if __name__ == "__main__":
    create_structure()