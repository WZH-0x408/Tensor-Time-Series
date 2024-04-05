import torch.nn as nn
import torch
'''
Define a simplest model as a template.
Please replace code to yours.
'''
class Template_Model(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=2) -> None:
        super(Template_Model, self).__init__()
        self.input_size = input_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = x.view(-1,self.input_size)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x