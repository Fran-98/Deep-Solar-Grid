import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, output_size=1):
        super().__init__()
        # 1. Añadimos bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, bidirectional=True)

        # 2. La capa de salida ahora debe aceptar el doble de features,
        #    porque la salida de la LSTM bidireccional concatena ambas direcciones.
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, input_seq):
        # El resto del forward es idéntico
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

###############
# LSTM con CNN 
###############
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super().__init__()
        # Capa convolucional que buscará patrones en ventanas de 3 pasos
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # La LSTM ahora recibe las features extraídas por la CNN
        self.lstm = nn.LSTM(64, hidden_size=hidden_layer_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, input_seq):
        # La CNN espera la forma (batch, channels/features, seq_len)
        # así que necesitamos permutar las dimensiones de la entrada
        x = input_seq.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        # Volvemos a permutar para la LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
    
###################
# LSTM con seq2seq
###################

class LSTM_Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_sequence_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, bidirectional=True)
        # La capa lineal ahora debe mapear la salida de la LSTM
        # al número de puntos que queremos predecir en el futuro.
        self.linear = nn.Linear(hidden_layer_size * 2, output_sequence_len)

    def forward(self, input_seq):
        # lstm_out contiene las salidas de la LSTM para cada paso de tiempo
        lstm_out, _ = self.lstm(input_seq)
        # Usamos la salida oculta del último paso de tiempo para hacer la predicción
        # de toda la secuencia futura.
        last_hidden_state = lstm_out[:, -1, :]
        predictions = self.linear(last_hidden_state)
        return predictions