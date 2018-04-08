import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # input channel is 1
        # output channel is 6
        # kernel shape is 5 * 5
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling window is 2 * 2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # Just put single scalar for square polling window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten spatial inputs
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        X = F.relu(self.fc2(x))
        x = self.fc3
        return x

    def num_flat_featurs(self, x):
        # Ignore batch dimension
        size = x.size()[1:]
        num_features = 1
        # Manually calculate flatten size
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    cnn_net = SimpleCNN()
    print(cnn_net)
    print(list(cnn_net.parameters()))
    print(len(list(cnn_net.parameters())))