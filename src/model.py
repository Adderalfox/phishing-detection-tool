import torch.nn as nn

class PhishModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_classes=2, maxlen=200, lstm_hidden=256, lstm_layers=2):
        super(PhishModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]

        x = self.fc_layers(x)
        return x
