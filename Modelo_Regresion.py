# %%
# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import pickle
import joblib

# Para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# Para preprocesamiento y construcción del modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score
from xgboost import XGBClassifier


# Cargarmos el conjunto de datos
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Realizamos una exploración inicial de los datos
print("--- Primeras 5 filas del conjunto de datos ---")
print(data.head())

print("\nDescripción estadística:")
print(data.describe())

print("\nDistribución de la variable objetivo:")
print(data['DEATH_EVENT'].value_counts())

# Verificamos si existen valores nulos
print("\nValores nulos en cada columna:")
print(data.isnull().sum())

print("\n--- Información General y Tipos de Datos ---")
print(data.info()) 
# Hallazgo: Se verifica que no hay datos nulos, ya que si los hubiera, el modelo fallaría.

data.head()
data.DEATH_EVENT.value_counts(normalize = True)

# --- Balance de la Clase Objetivo ---
# Realizamos un gráfico para ver como se distribuye la variable Objetivo
plt.figure(figsize=(4,3))
sns.countplot(x='DEATH_EVENT', data=data)
plt.title('Distribución de Fallecimientos (0: No, 1: Sí)')
plt.xlabel('DEATH_EVENT')
plt.ylabel('Número de pacientes')
plt.show()
# Si hay muchos más 0 que 1, el modelo podría predecir siempre 0. Se analiza si el desbalance es crítico.

#Seleccionamos las variables numéricas del data frame
num_vars = [
    'age',
    'ejection_fraction',
    'serum_creatinine',
    'serum_sodium',
    'time'
]

#Desplegamos gráficos con la distribución de las variables numéricas
for var in num_vars:
    plt.figure(figsize=(4,3))
    sns.histplot(data[var], bins=30, kde=True)
    plt.title(f'Distribución de {var}')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    plt.show()

# --- ANÁLISIS VISUAL: Boxplots (Variables Críticas) ---
#Gráficos de box plot para las variables numéricas separados por evento de muerte (paciente fallecido y sobreviviente)
for var in ['ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']:
    plt.figure(figsize=(4,3))
    sns.boxplot(x='DEATH_EVENT', y=var, data=data)
    plt.title(f'{var} por Evento de Muerte')
    plt.xlabel('DEATH_EVENT')
    plt.ylabel(var)
    plt.show()
#Los boxplots revelan outliers y si las medianas de los grupos son realmente distintas.

#Variables binarias por evento de muerte (paciente fallecido y sobreviviente)
bin_vars = [
    'anaemia',
    'diabetes',
    'high_blood_pressure',
    'smoking'
]

#Comparamos el riesgo relativo de las variables que son binarias
death_rates = data.groupby(bin_vars)['DEATH_EVENT'].mean().reset_index()

for var in bin_vars:
    plt.figure(figsize=(4,3))
    sns.barplot(x=var, y='DEATH_EVENT', data=data, estimator=np.mean)
    plt.title(f'Proporción de muerte según {var}')
    plt.ylabel('Proporción de DEATH_EVENT = 1')
    plt.show()

# Los gráficos anteriores muestran que hay una mayor proporción de muerte en pacientes con anemia y presion alta que en pacientes sin esas dos condiciones. El fumar y tener diabetes muestran proporciones muy similares. En este sentido, las mayores probabilidades condicionadas de muerte se observan en pacientes con anemia y con hipertensión.

#Matriz de correlación
corr_vars = [
    'age',
    'ejection_fraction',
    'serum_creatinine',
    'serum_sodium',
    'platelets',
    'creatinine_phosphokinase',
    'time',
    'DEATH_EVENT'
]

# --- Matriz de Correlación ---
corr = data[corr_vars].corr(method='pearson')

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Matriz de Correlación')
plt.show()
#Se busca qué variables tienen mayor correlación con DEATH_EVENT. Por ejemplo, 'time' (tiempo de seguimiento) tiene una correlación negativa alta.
#Aunque las variables presentan asociaciones individuales moderadas con el evento de muerte (menor a 0,3 a excepcion de time, que es la variable de seguimiento de los pacientes), cada una representa un factor de riesgo clínico distinto (la correlación es baja también entre variables predictivas, por lo cual no habría redundancia). En conjunto, estas variables aportan información complementaria que permite al modelo capturar patrones multivariados relevantes. Dado que no se observó multicolinealidad significativa entre los predictores y que el modelo muestra un desempeño global adecuado, se justifica el uso de todas las variables disponibles.

# --- Preparación de datos ---
# Separamos características y variable objetivo
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Dividimos los datos en conjuntos de entrenamiento y prueba con estratificación debido al desbalance. División Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Escalamos las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamos un modelo de Regresión Logística con balanceo de clases
lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Realizamos predicciones
y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]

# Evaluamos el modelo
print("\nReporte de clasificación para Regresión Logística:")
print(classification_report(y_test, y_pred))

# --- ANÁLISIS VISUAL: Matriz de Confusión ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Regresión Logística')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Calcular y mostrar el ROC AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC Score para Regresión Logística:", roc_auc)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label='Regresión Logística (área = %0.3f)' % roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Regresión Logística')
plt.legend()
plt.show()

# Curva Precision-Recall
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label='Regresión Logística')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.title('Curva Precision-Recall - Regresión Logística')
plt.legend()
plt.show()

#Guardamos el modelo entrenado
with open('modelo_regresion_logistica.pkl', 'wb') as archivo_salida:
    pickle.dump(lr, archivo_salida)

# Guardamos el scaler de los datos
with open('scaler.pkl', 'wb') as archivo_salida:
    pickle.dump(scaler, archivo_salida)

print("\n--- PROCESO COMPLETADO ---")
print("Modelo y análisis generados con éxito.")