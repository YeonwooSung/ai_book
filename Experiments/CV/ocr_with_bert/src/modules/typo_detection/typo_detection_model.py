import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))


class TypoDetectorBERT(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        self.mish_fn = Mish()
        self.dropout = nn.Dropout(0.2)
        
        self.kernel_1 = 2
        self.kernel_2 = 4
        self.embedding_dim = 768
        self.out_size = 32
        self.num_labels = 2
        
        self.conv_1 = nn.Conv1d(self.embedding_dim, self.out_size, self.kernel_1, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_1.weight, mode='fan_out')
        self.pool_1 = nn.MaxPool1d(kernel_size=self.kernel_1, stride=1)
        
        self.conv_2 = nn.Conv1d(self.embedding_dim, self.out_size, self.kernel_2, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_2.weight, mode='fan_out')
        self.pool_2 = nn.MaxPool1d(kernel_size=self.kernel_2, stride=1, padding=2)
        
        self.classifier = nn.Linear(self.out_size * 2, self.num_labels)  
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')
        
    def forward(self, input, attention_mask):
        out = self.bert(input, attention_mask)
        
        conv_inp = out[0].permute(0, 2, 1)
        
        conv_out_1 = self.pool_1(self.mish_fn(self.conv_1(conv_inp)))
        conv_out_2 = self.pool_2(self.mish_fn(self.conv_2(conv_inp)))
        
        conv_out = torch.cat([conv_out_1, conv_out_2],
                             axis = 1)
        conv_out = self.dropout(conv_out.permute(0, 2, 1))
        logits = self.classifier(conv_out)
        return logits


        