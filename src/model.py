import torch.nn as nn

class PhishCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2, maxlen=200):
        super(PhishCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        
        self.fc_layers = nn.Sequential(
            nn.Linear((maxlen // 4) * 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
