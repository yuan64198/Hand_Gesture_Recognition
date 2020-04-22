from torch import nn

class Gest_CNN(nn.Module):
    def __init__(self):
        super(Gest_CNN, self).__init__()
        self.features = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv1d(3, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv1d(64, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv1d(64, 64, kernel_size=5),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2496, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 20),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2496)
        out = self.classifier(x)
        return out