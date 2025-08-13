import torch

def weighted_mse_loss(input, target, weight_factor=15.0):
    # Creamos un peso para cada punto de datos.
    # Los pesos son más altos cuando el valor real (target) es más alto.
    weights = torch.ones_like(target) + target * weight_factor
    
    # Calculamos el error cuadrático normal
    loss = (input - target) ** 2
    
    # Aplicamos los pesos
    weighted_loss = loss * weights
    
    # Devolvemos la media
    return torch.mean(weighted_loss)

def zero_focused_loss(input, target, zero_penalty=10.0):
    # Error cuadrático normal
    loss = (input - target) ** 2

    # Máscara para encontrar dónde el valor real es casi cero
    zero_mask = (target < 0.01) # Usamos un valor pequeño por la escala [0,1]

    # Si hay valores cero en el batch, calcula la penalización
    if zero_mask.sum() > 0:
        # Penaliza las predicciones que no son cero en esos puntos
        penalty = (input[zero_mask] ** 2) * zero_penalty
        return torch.mean(loss) + torch.mean(penalty)
    else:
        return torch.mean(loss)