import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    # in: hour (1-24), day (1-7), week (1-52), year (>0), wind, consumption, temp, normal temp, and spot price at t-1
    # out: spot price at t
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, output_size=1):
        super(LSTM_model, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            batch_first=True) #Makes the input shape = (batch_size, sequence_length, input_size)
        
        self.linear = nn.Linear(in_features=hidden_size,
                               out_features=output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out[:, -1, :]) # lstm_out has shape (batch_size, sequence_length, hidden_size), so doing [:, -1, :] just says return entire batch, the LAST time step and all hidden features
        return output

