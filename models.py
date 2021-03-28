import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn
from transformers import BertModel


class BiEncoder(nn.Module):
    def __init__(
        self,
        share_params=True,
        checkpoint_path_a = None,
        checkpoint_path_b = None,
        pretrained_model_name = 'hfl/chinese-bert-wwm'
    ):
        super(BiEncoder, self).__init__()
        self.share_params = share_params
        self.checkpoint_path_a = checkpoint_path_a
        self.checkpoint_path_b = checkpoint_path_b

        self.model_a = BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        self.model_b = BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)

        if share_params:
            self.model_a = self.model_b

    def forward_once(self, ids, mask, token_type_ids):
        _ , output= self.model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        return output

    def forward(self, ids, mask, token_type_ids):
        output1 = self.forward_once(ids[0],mask[0], token_type_ids[0])
        output2 = self.forward_once(ids[1],mask[1], token_type_ids[1])
        return output1 , output2
    



# class SingelBertModel(nn.Module):
#     def __init__(
#         self,
#         checkpoint_path = None,
#         pretrained_model_name = 'hfl/chinese-bert-wwm'
#     ):
#         super(BertModel, self).__init__()
#         self.checkpoint_path = checkpoint_path_b

#         self.model_a = BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
#         self.model_b = BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)

#         if share_params:
#             self.model_a = self.model_b

#     def forward_once(self, ids, mask, token_type_ids):
#         _ , output= self.model(ids, attention_mask = mask, token_type_ids = token_type_ids)
#         return output

#     def forward(self, ids, mask, token_type_ids):
#         output1 = self.forward_once(ids[0],mask[0], token_type_ids[0])
#         output2 = self.forward_once(ids[1],mask[1], token_type_ids[1])
#         return output1 , output2