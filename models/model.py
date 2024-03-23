import torch
import torch.nn as nn

# Define a simple RNN with one layer and one neuron
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size

        # Weights for input, hidden state, and output
        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = torch.tanh(self.W_h(combined))
        output = self.W_out(hidden)
        return output, hidden