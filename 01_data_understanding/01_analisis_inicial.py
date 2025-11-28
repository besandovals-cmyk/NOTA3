import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración visual
sns.set_style('whitegrid')

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE RUTAS (Crucial para evitar errores)
# ---------------------------------------------------------
# Obtenemos la ruta absoluta de donde está este script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas a los archivos de datos (usando script_dir como ancla)
path_application = os.path.join(script_dir, '..', 'data', 'application_.parquet')
path_bureau = os.path.join(script_dir, '..', 'data', 'bureau.parquet')

# ---------------------------------------------------------
# 2. ANÁLISIS DE LA TABLA PRINCIPAL (Application)
# ---------------------------------------------------------
print("\n--- 1. Cargando Application Train ---")
try:
    df_train = pd.read_parquet(path_application, engine='pyarrow')
    print(f"✔ Datos cargados. Dimensiones: {df_train.shape}")
    
    if 'TARGET' in df_train.columns:
        # Análisis de Desbalance
        target_counts = df_train['TARGET'].value_counts(normalize=True) * 100
        print("\nDistribución del Target (%):")
        print(target_counts)
        
        # Gráfico (Corregido el warning de palette)
        plt.figure(figsize=(6,4))
        sns.countplot(x='TARGET', hue='TARGET', data=df_train, palette='viridis', legend=False)
        plt.title('Distribución de Riesgo (0: Paga, 1: No Paga)')
        plt.show()
    else:
        print("⚠️ Alerta: No se encontró la columna TARGET.")

except FileNotFoundError:
    print(f"❌ Error: No se encuentra {path_application}")
except Exception as e:
    print(f"❌ Error inesperado en Application: {e}")

# ---------------------------------------------------------
# 3. ANÁLISIS DE RELACIONES (Bureau)
# ---------------------------------------------------------
print("\n--- 2. Cargando Bureau (Historial Externo) ---")
try:
    df_bureau = pd.read_parquet(path_bureau, engine='pyarrow')
    print(f"✔ Bureau cargado. Dimensiones: {df_bureau.shape}")
    
    # Agrupamos por Cliente para ver cuántos créditos tiene cada uno
    bureau_counts = df_bureau.groupby('SK_ID_CURR').size()
    
    print("\nEstadísticas de Créditos Previos por Cliente:")
    print(bureau_counts.describe())
    
    # Gráfico de distribución
    plt.figure(figsize=(8,4))
    sns.histplot(bureau_counts, bins=50, kde=False, color='salmon')
    plt.title('Cantidad de créditos previos por cliente (Bureau)')
    plt.xlabel('Número de créditos')
    plt.xlim(0, 40)
    plt.show()

except FileNotFoundError:
    print(f"❌ Error: No se encuentra {path_bureau}")
except Exception as e:
    print(f"❌ Error en Bureau: {e}")