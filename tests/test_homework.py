# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import numpy as np
import os
import gzip
import json
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def cargar_y_limpiar(path):
    """
    Carga y limpia el dataset desde la ruta especificada.

    La limpieza incluye:
    - Renombrar la columna objetivo a 'default'.
    - Eliminar la columna 'ID'.
    - Eliminar registros con información no disponible. En este dataset,
      los valores 0 en 'EDUCATION' y 'MARRIAGE' se consideran N/A.
    - Agrupar los niveles de educación superiores (>4) en la categoría 4 ('others').
    """
    print(f"Cargando archivo: {path}")
    
    # Verificar que el archivo existe
    if not os.path.exists(path):
        print(f"ERROR: Archivo no encontrado - {path}")
        # Buscar archivos alternativos
        possible_files = [
            path.replace('.zip', ''),
            path.replace('.csv.zip', '.csv'),
            path.replace('train_data.csv.zip', 'train.csv'),
            path.replace('test_data.csv.zip', 'test.csv')
        ]
        
        for alt_path in possible_files:
            if os.path.exists(alt_path):
                print(f"Usando archivo alternativo: {alt_path}")
                path = alt_path
                break
        else:
            raise FileNotFoundError(f"No se encontró ningún archivo de datos en: {path}")
    
    # Cargar archivo con o sin compresión
    try:
        if path.endswith('.zip'):
            df = pd.read_csv(path, compression='zip')
        else:
            df = pd.read_csv(path)
        print(f"Archivo cargado exitosamente: {df.shape}")
    except Exception as e:
        print(f"Error cargando archivo: {e}")
        raise
    
    # Mostrar información del dataset original
    print(f"Columnas originales: {list(df.columns)}")
    print(f"Dimensiones originales: {df.shape}")
    
    # Renombrar columna objetivo
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
        print("✓ Columna objetivo renombrada a 'default'")
    elif "default" not in df.columns:
        print("WARNING: No se encontró la columna objetivo")
    
    # Eliminar columna ID si existe
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
        print("✓ Columna ID eliminada")
    
    # Mostrar información antes de limpiar
    print(f"Valores únicos en EDUCATION: {sorted(df['EDUCATION'].unique())}")
    print(f"Valores únicos en MARRIAGE: {sorted(df['MARRIAGE'].unique())}")
    
    # Eliminar registros con información no disponible
    # Los valores 0 en EDUCATION y MARRIAGE significan 'N/A'
    antes_limpieza = len(df)
    df = df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
    print(f"✓ Registros eliminados por EDUCATION=0 o MARRIAGE=0: {antes_limpieza - len(df)}")
    
    # Agrupar valores de EDUCATION > 4 en la categoría 4 ('others')
    antes_agrupacion = df['EDUCATION'].value_counts().sort_index()
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    despues_agrupacion = df['EDUCATION'].value_counts().sort_index()
    print(f"✓ Valores EDUCATION agrupados:")
    print(f"  Antes: {dict(antes_agrupacion)}")
    print(f"  Después: {dict(despues_agrupacion)}")
    
    # Eliminar cualquier valor NaN restante
    antes_nan = len(df)
    df = df.dropna()
    print(f"✓ Registros eliminados por NaN: {antes_nan - len(df)}")
    
    print(f"Dimensiones finales: {df.shape}")
    
    # Verificar que tenemos la columna objetivo
    if 'default' not in df.columns:
        raise ValueError("No se encontró la columna 'default' en el dataset")
    
    # Mostrar distribución de la variable objetivo
    print(f"Distribución de la variable objetivo:")
    print(df['default'].value_counts().sort_index())
    
    return df


def dividir_xy(df):
    """Divide el DataFrame en características (X) y variable objetivo (y)."""
    X = df.drop(columns=["default"])
    y = df["default"]
    print(f"Características (X): {X.shape}")
    print(f"Variable objetivo (y): {y.shape}")
    return X, y


def crear_modelo():
    """
    Crea un pipeline de preprocesamiento y modelado con GridSearchCV
    para la optimización de hiperparámetros.
    """
    print("Creando pipeline de ML...")
    
    # Variables categóricas para one-hot encoding
    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
    
    # Variables numéricas
    num_features = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    
    # Preprocessor para variables categóricas y numéricas
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features),
        ("num", StandardScaler(), num_features)
    ])
    
    # Pipeline con preprocessor y clasificador
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    # Parámetros más amplios para mejorar el rendimiento
    param_grid = {
        "classifier__n_estimators": [200, 30],
        "classifier__max_depth": [10, 15, 20, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ['sqrt', 'log2', None]
    }
    
    print(f"Parámetros a optimizar: {param_grid}")
    
    # GridSearchCV con validación cruzada
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        verbose=1
    )
    
    return grid


def guardar_modelo(model, ruta):
    """Guarda el modelo entrenado en la ruta especificada, comprimido con gzip."""
    print(f"Guardando modelo en: {ruta}")
    
    try:
        # Crear directorio si no existe
        dir_path = os.path.dirname(ruta)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Directorio creado/verificado: {dir_path}")
        
        # Guardar modelo comprimido
        with gzip.open(ruta, "wb") as f:
            pickle.dump(model, f)
        
        # Verificar que el archivo fue creado correctamente
        if os.path.exists(ruta):
            size = os.path.getsize(ruta)
            print(f"✓ Modelo guardado exitosamente: {ruta} ({size} bytes)")
        else:
            raise FileNotFoundError(f"El archivo no fue creado: {ruta}")
            
    except Exception as e:
        print(f"✗ Error guardando modelo: {e}")
        raise


def guardar_metricas(model, x_train, y_train, x_test, y_test, ruta):
    """
    Calcula y guarda las métricas de evaluación y las matrices de confusión
    para los conjuntos de entrenamiento y prueba en un archivo JSON.
    """
    print(f"Calculando métricas y guardando en: {ruta}")
    
    try:
        # Crear directorio si no existe
        dir_path = os.path.dirname(ruta)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Directorio creado/verificado: {dir_path}")
        
        resultados = []

        # Calcular métricas para train y test
        for x, y, name in [(x_train, y_train, "train"), (x_test, y_test, "test")]:
            print(f"Calculando métricas para {name}...")
            
            # Hacer predicciones
            y_pred = model.predict(x)
            
            # Calcular métricas
            precision = precision_score(y, y_pred, average='binary')
            balanced_acc = balanced_accuracy_score(y, y_pred)
            recall = recall_score(y, y_pred, average='binary')
            f1 = f1_score(y, y_pred, average='binary')
            
            print(f"  Precision: {precision:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            # Agregar métricas
            resultados.append({
                "type": "metrics",
                "dataset": name,
                "precision": float(precision),
                "balanced_accuracy": float(balanced_acc),
                "recall": float(recall),
                "f1_score": float(f1),
            })

            # Calcular matriz de confusión
            cm = confusion_matrix(y, y_pred)
            print(f"  Matriz de confusión:")
            print(f"    {cm}")
            
            # Agregar matriz de confusión (formato corregido)
            resultados.append({
                "type": "cm_matrix",
                "dataset": name,
                "true_0": {
                    "predicted_0": int(cm[0][0]),
                    "predicted_1": int(cm[0][1])
                },
                "true_1": {
                    "predicted_0": int(cm[1][0]),
                    "predicted_1": int(cm[1][1])
                }
            })

        # Guardar resultados en archivo JSON
        with open(ruta, "w", encoding="utf-8") as f:
            for linea in resultados:
                json.dump(linea, f, ensure_ascii=False)
                f.write("\n")
        
        print(f"✓ Métricas guardadas exitosamente en: {ruta}")
        
    except Exception as e:
        print(f"✗ Error guardando métricas: {e}")
        raise


def main():
    """
    Función principal que orquesta la carga de datos, entrenamiento del modelo,
    y guardado de resultados.
    """
    print("=== INICIANDO PROCESO DE MACHINE LEARNING ===")
    
    try:
        # Verificar estructura de directorios
        print("\n1. Verificando estructura de directorios...")
        print(f"Directorio actual: {os.getcwd()}")
        
        if os.path.exists("files/input"):
            print(f"Contenido de files/input: {os.listdir('files/input')}")
        else:
            print("ERROR: No existe el directorio files/input")
            return
        
        # Cargar y limpiar datos
        print("\n2. Cargando y limpiando datos...")
        
        # Intentar diferentes nombres de archivos
        train_files = [
            "files/input/train_data.csv.zip",
            "files/input/train_data.csv",
            "files/input/train.csv.zip",
            "files/input/train.csv"
        ]
        
        test_files = [
            "files/input/test_data.csv.zip",
            "files/input/test_data.csv",
            "files/input/test.csv.zip",
            "files/input/test.csv"
        ]
        
        train_file = None
        test_file = None
        
        for f in train_files:
            if os.path.exists(f):
                train_file = f
                break
        
        for f in test_files:
            if os.path.exists(f):
                test_file = f
                break
        
        if not train_file or not test_file:
            print("ERROR: No se encontraron archivos de datos")
            return
        
        print(f"Usando archivo de entrenamiento: {train_file}")
        print(f"Usando archivo de prueba: {test_file}")
        
        # Cargar datos
        train = cargar_y_limpiar(train_file)
        test = cargar_y_limpiar(test_file)
        
        # Dividir en X e y
        print("\n3. Dividiendo datos en características y variable objetivo...")
        x_train, y_train = dividir_xy(train)
        x_test, y_test = dividir_xy(test)

        # Crear y entrenar modelo
        print("\n4. Creando y entrenando modelo...")
        modelo = crear_modelo()
        
        print("Iniciando entrenamiento (esto puede tomar varios minutos)...")
        modelo.fit(x_train, y_train)
        
        print(f"✓ Entrenamiento completado!")
        print(f"Mejores parámetros: {modelo.best_params_}")
        print(f"Mejor score (balanced accuracy): {modelo.best_score_:.4f}")

        # Guardar modelo
        print("\n5. Guardando modelo...")
        guardar_modelo(modelo, "files/models/model.pkl.gz")
        
        # Guardar métricas
        print("\n6. Calculando y guardando métricas...")
        guardar_metricas(modelo.best_estimator_, x_train, y_train, x_test, y_test, "files/output/metrics.json")
        
        print("\n=== PROCESO COMPLETADO EXITOSAMENTE ===")
        
        # Verificación final
        print("\n7. Verificación final de archivos...")
        model_path = "files/models/model.pkl.gz"
        metrics_path = "files/output/metrics.json"
        
        if os.path.exists(model_path):
            print(f"✓ Modelo guardado: {model_path} ({os.path.getsize(model_path)} bytes)")
        else:
            print(f"✗ ERROR: No se encontró el modelo en {model_path}")
            
        if os.path.exists(metrics_path):
            print(f"✓ Métricas guardadas: {metrics_path} ({os.path.getsize(metrics_path)} bytes)")
        else:
            print(f"✗ ERROR: No se encontraron las métricas en {metrics_path}")
        
    except Exception as e:
        print(f"\n✗ ERROR EN EL PROCESO PRINCIPAL: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()