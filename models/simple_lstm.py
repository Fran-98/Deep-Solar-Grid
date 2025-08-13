import torch
import numpy as np

import matplotlib.pyplot as plt
import os
from loss import weighted_mse_loss, zero_focused_loss
from preprocess import Dataset
from models_nn import LSTMModel

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print(f"Usando dispositivo: {device}")

train_path = 'dataset_final_train.csv'
test_path = 'dataset_final_test.csv'

ruta_modelo = 'saved_models/lstm_multivariado.pth'

dataset = Dataset(train_path, test_path, n_pasos = 3)

# Convertir a tensores de PyTorch y moverlos al dispositivo seleccionado (CUDA/CPU/XPU)
X_train_tensor = torch.from_numpy(dataset.X_train).float().to(device)
y_train_tensor = torch.from_numpy(dataset.y_train).float().reshape(-1, 1).to(device)
X_test_tensor = torch.from_numpy(dataset.X_test).float().to(device)
y_test_tensor = torch.from_numpy(dataset.y_test).float().reshape(-1, 1).to(device)

print(f"\nForma del tensor X_train: {X_train_tensor.shape}")
print(f"Forma del tensor y_train: {y_train_tensor.shape}")

POTENCIA_MAXIMA = dataset.POTENCIA_MAXIMA

    
# --- 4. ENTRENAMIENTO ---
model = LSTMModel(input_size=dataset.n_features, hidden_layer_size=128, output_size=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("\nEstructura del modelo:")
print(model)

epochs = 2000
print("\nIniciando entrenamiento...")
for i in range(epochs):
    model.train()
    y_pred = model(X_train_tensor)
    # loss = weighted_mse_loss(y_pred, y_train_tensor)
    loss = zero_focused_loss(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 10 == 0:
        print(f'Epoch {i+1}/{epochs}, Loss: {loss.item():.6f}')

print("Entrenamiento finalizado.")

# --- 5. GUARDADO DEL MODELO ---
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')


torch.save(model.state_dict(), ruta_modelo)
print(f"\nModelo guardado en: {ruta_modelo}")

# --- EVALUACIÓN Y VISUALIZACIÓN ---
model.eval() # <-- Buena práctica: poner el modelo en modo evaluación
with torch.no_grad():
    test_predict = model(X_test_tensor)

# <-- IMPORTANTE: Mover las predicciones a la CPU antes de convertirlas a NumPy
test_predict_np = test_predict.cpu().numpy()

# Para invertir la transformación del scaler, necesitamos "engañarlo"
# Creando un array con la misma forma que los datos originales (n_features)
# y poniendo nuestras predicciones en la primera columna.
test_predict_padded = np.zeros((len(test_predict_np), dataset.n_features))
test_predict_padded[:, 0] = test_predict_np.flatten()
test_predict_orig = dataset.scaler.inverse_transform(test_predict_padded)[:, 0]

# Hacemos lo mismo para los valores reales de prueba
y_test_padded = np.zeros((len(dataset.y_test), dataset.n_features))
y_test_padded[:, 0] = dataset.y_test.flatten()
y_test_orig = dataset.scaler.inverse_transform(y_test_padded)[:, 0]

# Visualización (sin cambios)
plot_index = dataset.df_test.index[dataset.n_pasos:]
plt.figure(figsize=(15, 6))
plt.plot(plot_index, y_test_orig, label='Valores Reales (Prueba)')
plt.plot(plot_index, test_predict_orig, label='Predicción (Prueba)', alpha=0.7)
plt.title('Predicción Multivariada de Potencia Activa con LSTM')
plt.xlabel('Fecha')
plt.ylabel('Potencia activa (kW)')
plt.legend()
plt.grid(True)
plt.show()