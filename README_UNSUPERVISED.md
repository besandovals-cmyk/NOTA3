# EA3 Práctico: Análisis No Supervisado (Clustering)

Este documento describe la implementación de técnicas de aprendizaje no supervisado para complementar el modelo de Scoring Crediticio, cumpliendo con los requerimientos de la evaluación EA3.

## 1. Técnica Seleccionada: K-Means Clustering
Se eligió el algoritmo **K-Means** para segmentar a los clientes en grupos homogéneos basados en su comportamiento financiero, sin utilizar la etiqueta de "Default".

* **Justificación:** El objetivo es descubrir si existen perfiles de clientes "naturales" (ej. perfiles conservadores vs. arriesgados) y evaluar si estos grupos correlacionan con el riesgo de impago. Esto permite validar si el modelo supervisado está capturando patrones de comportamiento reales.

## 2. Instrucciones de Ejecución
El código se encuentra aislado en la carpeta `ea3_unsupervised` para evitar fugas de datos (Data Leakage) con el modelo principal.

**Ejecutar el análisis:**
```bash
python ea3_unsupervised/analisis_clusters.py
```
Salidas generadas:

`ea3_unsupervised/graficos/`: Contiene la visualización de los clusters (PCA 2D).

`ea3_unsupervised/resultados_ea3.txt`: Resumen estadístico del riesgo por cluster.

## 3. Análisis e Interpretación de Resultados
Se aplicó K-Means (k=3) sobre las variables numéricas del set de entrenamiento, previamente escaladas. Posteriormente, se cruzaron los clusters obtenidos con la variable objetivo real (TARGET) para analizar la tasa de incumplimiento en cada grupo.

Hallazgos: El algoritmo logró diferenciar grupos con niveles de riesgo notablemente distintos:

Cluster 0: Tasa de incumplimiento del 5.7%. (Perfil: Riesgo Bajo/Medio)

Cluster 1: Tasa de incumplimiento del 10.0%. (Perfil: Riesgo Bajo/Medio)

Cluster 2: Tasa de incumplimiento del 4.9%. (Perfil: Riesgo Alto)

Nota: La tasa global promedio de incumplimiento en el dataset es ~8.1%.

Interpretación: El hecho de que un algoritmo no supervisado (que no conoce quién pagó y quién no) haya logrado aislar un grupo (Cluster 2) con una tasa de incumplimiento mucho mayor al promedio sugiere que existen patrones estructurales claros en los datos financieros de estos clientes.

## 4. Discusión: Integración al Proyecto Final
¿Es útil esta técnica para el modelo final? SÍ.

Razones:

Nueva Característica (Feature): La etiqueta del cluster (Cluster_ID) puede agregarse como una variable categórica extra al modelo supervisado (LightGBM). Esto le daría al modelo una "pista" macro sobre el perfil del cliente.

Explicabilidad: Permite al negocio entender no solo el score de riesgo, sino el "tipo de cliente" (ej. "Este cliente fue rechazado porque su perfil financiero se parece al del Grupo 2, que históricamente tiene alta mora").

------------

Autor: Vicente Vásquez Caro
