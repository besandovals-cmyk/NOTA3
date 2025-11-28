import pandas as pd
import os
import gc # Garbage Collector para liberar memoria RAM automáticamente

# ---------------------------------------------------------
# CONFIGURACIÓN Y RUTAS
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
output_dir = os.path.join(script_dir, '..', 'data') # Guardaremos el procesado aquí

def cargar_datos():
    print("⏳ Cargando datasets...")

    app_path = os.path.join(data_dir, 'application_.parquet')
    bureau_path = os.path.join(data_dir, 'bureau.parquet')
    
    if not os.path.exists(app_path) or not os.path.exists(bureau_path):
        raise FileNotFoundError(f"❌ Faltan archivos en {data_dir}. Verifica los nombres.")

    df_app = pd.read_parquet(app_path, engine='pyarrow')
    df_bureau = pd.read_parquet(bureau_path, engine='pyarrow')
    
    print(f"✔ Application cargada: {df_app.shape}")
    print(f"✔ Bureau cargada: {df_bureau.shape}")
    return df_app, df_bureau

def procesar_bureau(df_bureau):
    """
    Transforma la relación 'uno a muchos' (varios créditos por cliente) 
    en 'uno a uno' mediante agregaciones estadísticas (promedio, máximo, suma).
    """
    print("⚙️ Iniciando ingeniería de características en Bureau...")
    
    # 1. SEPARAR VARIABLES
    # Numéricas: Calcularemos estadísticas (media, max, suma)
    bureau_num = df_bureau.select_dtypes(include=['number']).columns.tolist()
    if 'SK_ID_CURR' in bureau_num: bureau_num.remove('SK_ID_CURR')
    
    # Categóricas: Calcularemos qué tan frecuentes son (One-Hot Encoding)
    bureau_cat = df_bureau.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. ONE-HOT ENCODING
    # Convertimos 'Active', 'Closed' en columnas numéricas (0 o 1)
    print("   - Codificando variables categóricas (One-Hot)...")
    df_bureau_enc = pd.get_dummies(df_bureau, columns=bureau_cat, dummy_na=True)
    
    # 3. DEFINIR REGLAS DE AGREGACIÓN
    agg_rules = {}
    # Para números: dame el promedio, el máximo y el mínimo
    for col in bureau_num:
        agg_rules[col] = ['mean', 'max', 'min', 'sum']
    
    # Para categorías codificadas: dame el promedio (que equivale al porcentaje de veces que aparece)
    cat_cols_enc = [c for c in df_bureau_enc.columns if c not in bureau_num and c != 'SK_ID_CURR']
    for col in cat_cols_enc:
        agg_rules[col] = ['mean']
        
    # 4. AGRUPAR POR CLIENTE (La operación clave del examen)
    print("   - Agrupando por cliente (Aggregation)...")
    bureau_agg = df_bureau_enc.groupby('SK_ID_CURR').agg(agg_rules)
    
    # 5. RENOMBRAR COLUMNAS
    # Las columnas quedan como ('DAYS_CREDIT', 'mean'). Las aplanamos a 'BURO_DAYS_CREDIT_MEAN'
    bureau_agg.columns = pd.Index([f'BURO_{c[0].upper()}_{c[1].upper()}' for c in bureau_agg.columns.tolist()])
    
    print(f"✔ Bureau procesado. Ahora tenemos {bureau_agg.shape[1]} nuevas características por cliente.")
    return bureau_agg

def main():
    # 1. Carga
    df_app, df_bureau = cargar_datos()
    
    # 2. Procesamiento (Feature Engineering)
    df_bureau_agg = procesar_bureau(df_bureau)
    
    # 3. Unión (Left Join)
    # Usamos left join para mantener todos los clientes de la tabla principal, 
    # incluso si no tienen historial en bureau (quedarán con nulos, que es correcto).
    print("🔗 Uniendo Tabla Maestra con Historial de Buró...")
    df_final = df_app.merge(df_bureau_agg, on='SK_ID_CURR', how='left')
    
    print(f"✨ Dataset Maestro Creado.")
    print(f"   Dimensiones Iniciales: {df_app.shape}")
    print(f"   Dimensiones Finales:   {df_final.shape}")
    
    # 4. Guardado
    output_path = os.path.join(output_dir, 'dataset_maestro.parquet')
    print(f"💾 Guardando archivo procesado en: {output_path}")
    df_final.to_parquet(output_path, engine='pyarrow')
    
    print("✅ Fase 1 completada con éxito.")

if __name__ == "__main__":
    main()