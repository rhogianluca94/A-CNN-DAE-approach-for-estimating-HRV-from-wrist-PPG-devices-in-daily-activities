import torch.nn as nn

class DAE_CNN(nn.Module):
    
    
    def __init__(self, kernel_size=16, dropoutRate = 0.0):
        super(DAE_CNN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=kernel_size, padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(32),
            nn.Dropout1d(dropoutRate),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=kernel_size, padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(dropoutRate),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 64, kernel_size=kernel_size, padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(dropoutRate),
            nn.MaxPool1d(kernel_size=2), 

            nn.Conv1d(64, 64, kernel_size=kernel_size, padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(dropoutRate),
            nn.MaxPool1d(kernel_size=2), 

            nn.Conv1d(64, 64, kernel_size=kernel_size, padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(dropoutRate),
            nn.MaxPool1d(kernel_size=2)
            
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout1d(dropoutRate),

            nn.Conv1d(128, 256, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 128, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 64, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 64, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        # Decoder (using ConvTranspose1d)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=0),
            nn.ELU(), 
            nn.BatchNorm1d(64),

            nn.ConvTranspose1d(64, 64, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
            nn.ELU(), 
            nn.BatchNorm1d(64),

            nn.ConvTranspose1d(64, 64, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=0),
            nn.ELU(), 
            nn.BatchNorm1d(64),

            nn.ConvTranspose1d(64, 32, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=0), 
            nn.ELU(), 
            nn.BatchNorm1d(32),

            nn.ConvTranspose1d(32, 1, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=0),
        )

        # Init the batch counter
        self.batch_counter = 0

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x