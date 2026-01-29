# Heart Failure Death Event Prediction API

## Descripción del problema

Las enfermedades cardiovasculares son una de las principales causas de muerte a nivel mundial. En particular, la insuficiencia cardíaca presenta una elevada tasa de mortalidad, lo que hace relevante contar con herramientas que permitan anticipar el riesgo de fallecimiento de los pacientes.

Este proyecto utiliza un conjunto de datos clínicos obtenido desde Kaggle (“Heart Failure Clinical Records Dataset”), el cual contiene información de pacientes con insuficiencia cardíaca, incluyendo variables demográficas, clínicas y de laboratorio.

El objetivo es construir un modelo de Machine Learning que permita predecir el evento `DEATH_EVENT` (0 = sobrevive, 1 = fallece) a partir de estas variables.  
El modelo se entrena de forma local, se guarda como archivos serializados (`model.pkl` y `scaler.pkl`) y posteriormente se expone mediante una API REST desarrollada con FastAPI, la cual es desplegada en la plataforma cloud **Google Cloud Platform (GCP)**.

---

## Plataforma cloud usada para el deploy

Para el despliegue de la API se utilizó la plataforma **Google Cloud Platform (GCP)**.

La aplicación se publica como un servicio web accesible mediante HTTP, permitiendo consultar el modelo de Machine Learning entrenado a través de una API REST construida con FastAPI.

El repositorio contiene los siguientes archivos necesarios para el despliegue en la nube:

- `app/main.py` (definición de la API con FastAPI)  
- `model.pkl` y `scaler.pkl` (modelo entrenado)  
- `requirements.txt` (dependencias del proyecto)  
- `runtime.txt` (versión de Python utilizada)  
- `Procfile` (comando de inicio del servicio)

---

## Descripción del dataset

El dataset “Heart Failure Clinical Records Dataset” contiene registros médicos de pacientes con insuficiencia cardíaca e incluye 12 variables predictoras y una variable objetivo (`DEATH_EVENT`).

Las variables disponibles en el conjunto de datos son:

- `age`: edad del paciente  
- `anaemia`: presencia de anemia (0 = no, 1 = sí)  
- `creatinine_phosphokinase`: nivel de creatinina fosfoquinasa en la sangre  
- `diabetes`: presencia de diabetes (0 = no, 1 = sí)  
- `ejection_fraction`: fracción de eyección del corazón  
- `high_blood_pressure`: presencia de hipertensión arterial (0 = no, 1 = sí)  
- `platelets`: recuento de plaquetas en la sangre  
- `serum_creatinine`: nivel de creatinina sérica  
- `serum_sodium`: nivel de sodio sérico  
- `sex`: sexo del paciente (0 = mujer, 1 = hombre)  
- `smoking`: hábito de fumar (0 = no, 1 = sí)  
- `time`: tiempo de seguimiento en días  
- `DEATH_EVENT`: variable objetivo (0 = sobrevive, 1 = fallece)

---

## Instrucciones para correr la API localmente (Windows / PowerShell)

### 1) Crear y activar entorno virtual

```powershell
python -m venv MDS_CC
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\MDS_CC\Scripts\Activate.ps1
```

---

## Ejemplo de uso del endpoint `/predict`

La API expone un endpoint `/predict` que recibe los datos clínicos de un paciente en formato JSON y devuelve la predicción del modelo respecto a la probabilidad de fallecimiento.

### Input (ejemplo 1)
```Json
{
  "age": 65,
  "anaemia": 0,
  "creatinine_phosphokinase": 250,
  "diabetes": 1,
  "ejection_fraction": 35,
  "high_blood_pressure": 1,
  "platelets": 263000,
  "serum_creatinine": 1.3,
  "serum_sodium": 137,
  "sex": 1,
  "smoking": 0,
  "time": 120
}
```

### Output (ejemplo 1)

```Json
{
  "prediction": 1,
  "probability": 0.72
}
```
