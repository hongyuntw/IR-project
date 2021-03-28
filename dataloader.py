import torch
from torch.utils.data import Dataset
import csv

class LcqmcDataset(Dataset):
    def __init__(self, mode, dir_path , max_len , tokenizer):
        #  mode in  [train,dev,test]
        self.mode = mode
        self.max_len = max_len
        self.tokenizer = tokenizer
        # if self.mode == 'train' or self.mode == 'dev':
        tsv_file_path = dir_path + self.mode + '.tsv'
        tsv_file = open(tsv_file_path)
        self.data_rows = list(csv.reader(tsv_file, delimiter="\t"))[1:]
        tsv_file.close()
        # elif self.mode == 'test':

    def tokenize(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_tensors='pt',
            truncation=True
        )
        ids = inputs['input_ids'][0]
        mask = inputs['attention_mask'][0]
        token_type_ids = inputs["token_type_ids"][0]
        return ids, mask, token_type_ids
        


    def __getitem__(self, idx):
        text_a, text_b, label = self.data_rows[idx]
        label = int(label)
        idsa, maska, token_type_idsa = self.tokenize(text_a)
        idsb, maskb, token_type_idsb = self.tokenize(text_b)

        return {
            'ids': [idsa , idsb],
            'mask': [maska , maskb],
            'token_type_ids': [token_type_idsa, token_type_idsb],
            'label': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data_rows)


class TaipeiQADataset(Dataset):
    def __init__(self, mode, dir_path , max_len , tokenizer):
        #  mode in  [train,dev,test]
        self.mode = mode
        self.max_len = max_len
        self.tokenizer = tokenizer
        # if self.mode == 'train' or self.mode == 'dev':
        tsv_file_path = dir_path + self.mode + '.tsv'
        tsv_file = open(tsv_file_path)
        self.data_rows = list(csv.reader(tsv_file, delimiter="\t"))[1:]
        tsv_file.close()
        # elif self.mode == 'test':

    def tokenize(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_tensors='pt',
            truncation=True
        )
        ids = inputs['input_ids'][0]
        mask = inputs['attention_mask'][0]
        token_type_ids = inputs["token_type_ids"][0]
        return ids, mask, token_type_ids
        


    def __getitem__(self, idx):
        label, text = self.data_rows[idx]
        label = int(label)
        ids, mask, token_type_ids = self.tokenize(text)

        return {
            'ids': ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'label': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data_rows)