import numpy as np
import pandas as pd

# Configuración
np.random.seed(42)  # Para reproducibilidad
n_productos = 4651  # Número de productos en el dataset

# Generación de datos sintéticos
precio_unitario = np.round(np.random.uniform(10, 150, n_productos), 2)  # Precios entre 10 y 1000
cantidad_inventario = np.random.randint(1, 500, n_productos)  # Cantidad de 1 a 500
velocidad_rotacion = np.round(np.random.uniform(0.1, 30, n_productos), 1)  # Rotación entre 0.1 y 50 ventas por semana
volumen = np.round(np.random.uniform(0.01, 10, n_productos), 2)  # Tamaño entre 0.01 y 10 m³
costo_almacenamiento = np.round(volumen * cantidad_inventario, 2)  # Costo = Volumen * Cantidad en Inventario
peso = np.round(np.random.uniform(0.5, 15, n_productos), 2)

# Crear el DataFrame
dataset = pd.DataFrame({
    'Precio Unitario': precio_unitario,
    'Cantidad en Inventario': cantidad_inventario,
    'Velocidad de Rotación': velocidad_rotacion,
    'Volumen (m³)': volumen,
    'Costo de Almacenamiento': costo_almacenamiento,
    'peso': peso,
    
})

# Guardar el dataset en un archivo CSV (opcional)
dataset.to_csv('dataset_productos.csv', index=False)

# Mostrar las primeras filas del dataset
print(dataset.head())