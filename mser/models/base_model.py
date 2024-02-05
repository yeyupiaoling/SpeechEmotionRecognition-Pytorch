from torch import nn


class BaseModel(nn.Module):
    def __init__(self, input_size=768, num_class=4, hidden_size=256):
        super().__init__()
        self.relu = nn.ReLU()
        self.pre_net = nn.Linear(input_size, hidden_size)
        self.post_net = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x = self.relu(self.pre_net(x))
        x = self.post_net(x)
        return x
