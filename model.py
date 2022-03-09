from torch import nn
import torch


class lstm_model(nn.Module):
    def __init__(self, inputs_size, num_unit, outputs_size, num_layers, sequence_length, device):
        super(lstm_model, self).__init__()
        self.lstm = nn.LSTM(input_size=inputs_size, hidden_size=num_unit, dropout=0.2, batch_first=True, num_layers=num_layers)
        self.sequence_length = sequence_length
        self.fc = nn.Linear(self.sequence_length*num_unit, outputs_size)
        self.softmax = nn.Softmax()
        self.num_unit = num_unit
        self.num_layer = num_layers
        self.outputs_size = outputs_size
        self.device = device

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.size(0), self.num_unit, dtype=torch.float).to(self.device)
        c0 = torch.zeros(self.num_layer, x.size(0), self.num_unit, dtype=torch.float).to(self.device)
        x, (h_n, c_n)= self.lstm(x, (h0, c0))
        x = x.contiguous().view(x.size(0), self.sequence_length*self.num_unit)
        x = self.fc(x)
        x = self.softmax(x)
        return x


# if __name__ ==  '__main__':
#     input_size = 38
#     out_size = 38
#     seq_len = 64
#     num_layer =1
#     num_unit = 256
#     inputs_data = torch.randn(16,seq_len,input_size)
#     rnn = lstm_model(input_size,num_unit,out_size,num_layer,seq_len,device='cpu')
#     rnn.init_weight()
#     for name, param in rnn.lstm.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.0)
#             elif 'weight' in name:
#                 nn.init.xavier_uniform_(param, gain=1)
#                 print(param)