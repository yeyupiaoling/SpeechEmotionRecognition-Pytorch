from torch import nn


class BaseModel(nn.Module):
    def __init__(self, input_size=768, num_class=4):
        super().__init__()
        self.relu = nn.ReLU()
        self.pre_net = nn.Linear(input_size, 256)
        self.post_net = nn.Linear(256, num_class)

    def forward(self, x, padding_mask=None):
        x = self.relu(self.pre_net(x))
        x = x * (1 - padding_mask.unsqueeze(-1).float())
        x = x.sum(dim=1) / (1 - padding_mask.float()).sum(dim=1, keepdim=True)
        x = self.post_net(x)
        return x
