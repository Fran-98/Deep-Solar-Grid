import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

from loss import zero_focused_loss
from preprocess import Dataset
from models_nn import CNN_LSTM_Model, LSTM_Seq2Seq, LSTMModel

# --- 1. CONFIGURACIÓN ---
# Configuración del dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, 'xpu') and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
print(f"Usando dispositivo: {device}")

# Rutas y parámetros
train_path = 'dataset_final_train.csv'
test_path = 'dataset_final_test.csv'
ruta_guardado = 'saved_models'
nombre_modelo = 'lstm_simple_no_met'
ruta_modelo = os.path.join(ruta_guardado, f'{nombre_modelo}.pth')
seq2seq = 'seq2seq' in nombre_modelo
ruta_grafico_loss = os.path.join(ruta_guardado, f'{nombre_modelo}_loss_curve.png')

# --- 2. PREPARACIÓN DE DATOS ---
# Instanciamos la clase que procesa los datos
dataset = Dataset(train_path, test_path, n_pasos=3, seq2seq=seq2seq)

# --- NUEVO: Creación del Conjunto de Validación ---
# Dividimos los datos de entrenamiento en un nuevo conjunto de entrenamiento y uno de validación (80/20)
# Esto nos permite monitorear si el modelo está sobreajustando.
X_train_split, X_val, y_train_split, y_val = train_test_split(
    dataset.X_train, dataset.y_train, test_size=0.2, shuffle=False # shuffle=False es importante en series temporales
)

# Convertir todos los conjuntos a tensores de PyTorch y moverlos al dispositivo
if not seq2seq:
    X_train_tensor = torch.from_numpy(X_train_split).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_split).float().reshape(-1, 1).to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).float().reshape(-1, 1).to(device)
    X_test_tensor = torch.from_numpy(dataset.X_test).float().to(device)
    y_test_tensor = torch.from_numpy(dataset.y_test).float().reshape(-1, 1).to(device)
else:
    X_train_tensor = torch.from_numpy(X_train_split).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_split).float().to(device) # <-- SIN .reshape()
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).float().to(device) # <-- SIN .reshape()
    X_test_tensor = torch.from_numpy(dataset.X_test).float().to(device)
    y_test_tensor = torch.from_numpy(dataset.y_test).float().to(device) # <-- SIN .reshape()


print(f"\nForma del tensor X_train: {X_train_tensor.shape}")
print(f"Forma del tensor X_val:   {X_val_tensor.shape}")

# --- 3. INICIALIZACIÓN DEL MODELO Y OPTIMIZADOR ---
model = LSTMModel(input_size=dataset.n_features, hidden_layer_size=256, output_size=1).to(device)
# model = CNN_LSTM_Model(input_size=dataset.n_features, hidden_layer_size=256, output_size=1).to(device)
# model = LSTM_Seq2Seq(input_size=dataset.n_features, hidden_layer_size=256, output_sequence_len=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
# Opcional pero recomendado: Un scheduler para ajustar la tasa de aprendizaje
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

print("\nEstructura del modelo:")
print(model)

# --- 4. BUCLE DE ENTRENAMIENTO CON VALIDACIÓN ---
epochs = 5000
# NUEVO: Listas para guardar el historial de pérdidas
train_losses = []
val_losses = []

# --- Variables para guardar el mejor modelo ---
best_val_loss = float('inf') # Inicializamos la mejor pérdida de validación con un valor infinito
best_model_state = None # Variable para guardar el estado del mejor modelo

print("\nIniciando entrenamiento...")
for i in range(epochs):
    # --- Fase de Entrenamiento ---
    model.train()
    y_pred = model(X_train_tensor)
    train_loss = zero_focused_loss(y_pred, y_train_tensor, zero_penalty=20)
    
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    train_losses.append(train_loss.item())
    
    # --- Fase de Validación ---
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_loss = zero_focused_loss(val_pred, y_val_tensor)
        val_losses.append(val_loss.item())

    # --- Lógica para guardar el mejor modelo ---
    # Comprobamos si la pérdida de validación actual es mejor que la mejor hasta ahora
    # if val_loss.item() < best_val_loss:
    #     best_val_loss = val_loss.item()
    #     best_model_state = model.state_dict() # Guardamos una copia del estado del modelo
    #     print(f"Mejor modelo encontrado en la época {i+1} con una pérdida de validación de: {best_val_loss:.6f}")

    # Actualizamos el scheduler con la pérdida de validación
    scheduler.step(val_loss)
    
    if (i+1) % 50 == 0:
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict() # Guardamos una copia del estado del modelo
            print(f"Mejor modelo encontrado en la época {i+1} con una pérdida de validación de: {best_val_loss:.6f}")
        print(f'Epoch {i+1}/{epochs}, Train Loss: {train_loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

print("Entrenamiento finalizado.")

# --- 5. GUARDADO DEL MEJOR MODELO Y GRÁFICO DE PÉRDIDAS ---
if not os.path.exists(ruta_guardado):
    os.makedirs(ruta_guardado)

# Guardamos el mejor modelo fuera del bucle
if best_model_state is not None:
    torch.save(best_model_state, ruta_modelo)
    print(f"\nEl mejor modelo basado en validación se ha guardado en: {ruta_modelo}")
else:
    print("No se encontró un mejor modelo para guardar.")

# --- NUEVO: Crear y guardar el gráfico de pérdidas ---
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Pérdida de Entrenamiento')
plt.plot(val_losses, label='Pérdida de Validación')
plt.title('Curvas de Pérdida de Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Loss)')
plt.legend()
plt.grid(True)
plt.savefig(ruta_grafico_loss)
print(f"Gráfico de pérdidas guardado en: {ruta_grafico_loss}")
plt.show()

# --- 6. EVALUACIÓN FINAL SOBRE EL CONJUNTO DE TEST ---
# (Opcional) Cargamos el mejor modelo guardado para la evaluación
if os.path.exists(ruta_modelo):
    model.load_state_dict(torch.load(ruta_modelo))
    model.eval()
    with torch.no_grad():
        # ... (Tu código de evaluación y visualización del conjunto de test puede ir aquí si lo deseas) ...
        # Por ejemplo, puedes calcular la pérdida final en el conjunto de prueba
        test_pred = model(X_test_tensor)
        test_loss = zero_focused_loss(test_pred, y_test_tensor)
        print(f"Pérdida final en el conjunto de prueba (con el mejor modelo): {test_loss.item():.6f}")
