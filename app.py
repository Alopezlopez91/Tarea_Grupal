from fastapi import FastAPI, HTTPException # FastAPI para el servidor y HTTPException para manejar errores
from pydantic import BaseModel # Para definir la estructura (esquema) de los datos que recibiremos.
import pickle # Para cargar los archivos binarios (.pkl) del modelo y el escalador.
import numpy as np # Para manejo de arreglos numéricos.
import pandas as pd # Para convertir los datos recibidos en tablas (DataFrames) que el modelo entienda.

# --- CARGA DE ARTEFACTOS (Lo hacemos una sola vez al iniciar el servidor) ---
# Cargamos el modelo y el scaler desde los archivos .pkl
# Se abre el archivo del modelo en modo lectura binaria.
with open('modelo_regresion_logistica.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

# Se abre el archivo del escalador en modo lectura binaria.
with open('scaler.pkl', 'rb') as archivo_scaler:
    scaler = pickle.load(archivo_scaler) # Se carga el escalador para normalizar los datos entrantes.

# Lista con los nombres de las 12 columnas en el orden exacto que requiere el modelo.
columnas = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking','time']

# --- CONFIGURACIÓN DE LA APP ---
# Creamos la aplicación FastAPI
# Instanciamos FastAPI y le asignamos un título que aparecerá en la documentación automática.
app = FastAPI(title="Detección de muerte por insuficiencia cardíaca")

@app.get("/health")
async def health_check():
    """
    Verificamos que el servidor esté activo y que el modelo/scaler 
    estén cargados correctamente en memoria.
    """
    return {
        "status": "healthy",
        "model_loaded": modelo is not None,
        "scaler_loaded": scaler is not None
    }

# --- DEFINICIÓN DEL ESQUEMA ---
# Definimos el modelo de datos de entrada utilizando Pydantic
# Creamos una clase que hereda de BaseModel. Esto obliga a que los datos lleguen en el formato correcto.
class Transaccion(BaseModel):
    age: float
    anaemia: float
    creatinine_phosphokinase: float
    diabetes: float
    ejection_fraction: float
    high_blood_pressure: float
    platelets: float
    serum_creatinine: float
    serum_sodium: float
    sex: float
    smoking: float
    time: float # Todas las variables se definen como 'float' para validar que sean números.

# --- CREAMOS EL ENDPOINT ---
# Definimos el endpoint para predicción
# Definimos una ruta '/prediccion/' que acepta el método POST (envío de datos).
@app.post("/prediccion/")
async def predecir_muerte_por_insuficiencia_cardíaca(transaccion: Transaccion): # La función recibe un objeto tipo 'Transaccion'.
    try:
        # Paso 1: Convertimos el objeto JSON recibido en un DataFrame de Pandas.
        # transaccion.dict() convierte los datos en un diccionario de Python.
        # Convertimos la entrada en un DataFrame
        datos_entrada = pd.DataFrame([transaccion.dict()], columns=columnas)
        
        # Paso 2: Aplicamos el escalador cargado anteriormente.
        # Solo usamos transform(), no fit(), para mantener la escala original.
        # Escalamos las características
        datos_entrada_scaled = scaler.transform(datos_entrada)
        
        # Paso 3: El modelo realiza la predicción (0 o 1).
        # Realizar la predicción
        prediccion = modelo.predict(datos_entrada_scaled)
        
        # Paso 4: Calculamos la probabilidad de que pertenezca a la clase 1 (Muerte por insuficiencia cardíaca).
        probabilidad = modelo.predict_proba(datos_entrada_scaled)[:, 1]
        
        # Paso 5: Construimos un diccionario con la respuesta final.
        # Usamos bool() y float() para que los datos sean compatibles con el formato JSON de salida.
        # Construimos la respuesta
        resultado = {
            "¿Alta probabilidad de muerte por insuficiencia cardíaca?": bool(prediccion[0]),
            "Probabilidad de muerte por insuficiencia cardíaca": float(probabilidad[0])
        }
        
        return resultado # FastAPI envía este diccionario automáticamente como un JSON al cliente.
    except Exception as e:
        # Si ocurre cualquier error (datos nulos, falla de memoria, etc.),
        # enviamos un error 400 con el detalle de qué falló.
        raise HTTPException(status_code=400, detail=str(e))
