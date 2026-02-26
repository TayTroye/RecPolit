from tokenizer import AbstractTokenizer
import json
import os
import torch
import numpy as np
from utils import *

# 用于tmall数据集
class TraRecTokenizer(AbstractTokenizer):
    """
    Tokenizer for SASRec model.

    An example:
        0: padding
        1-n_items: item tokens
        n_items+1: eos token

    Attributes:
        item2tokens (dict): A dictionary mapping items to their internal IDs.
        eos_token (int): The end-of-sequence token.
        ignored_label (int): Should be -100. Used to ignore the loss for padding tokens in `transformers`.
    """
    def __init__(self, config, sem_id_epoch=None):

        self.sem_id_epoch = sem_id_epoch
        super(TraRecTokenizer, self).__init__(config)

        id_mapping_file = config["id_map_file"]
        id_mapping = load_json(id_mapping_file)
        self.item2tokens = id_mapping['item2id']


        self.ignored_label = -100
        self.tokens2item = {v: k for k, v in self.item2tokens.items()}

        self.item_num = len(self.item2tokens)

        self.click_token = self.item_num 
        self.cart_token  = self.item_num + 1
        self.collet_token = self.item_num + 2
        self.purchase_token = self.item_num + 3
        self.bos_token = self.item_num + 4   # Session Start
        self.eos_token = self.item_num + 5   # Session End

        self.stage = config["stage"]
        

        self.max_len = config['max_item_seq_len'] * 4  + 1 # bos,action,item,eos


    def _load_item2tokens(self):
        """
        In SASRec, we just use the atomic ID from id_mapping.
        This method is kept for compatibility if the parent calls it, 
        but we handled loading in __init__ for clarity.
        """
        # Already loaded in __init__, just return references
        return self.item2tokens, self.tokens2item


    def tokenize(self, example: dict, split: str) -> dict:
        item_seq = example["item_seq"]
        target_seq = example["target_seq"]
        max_len = self.max_len

        # ==========================================================
        # Train Split
        # ==========================================================
        if split == 'train':
            # Encoder Input (Source)
            full_seq = item_seq 
            if len(full_seq) > max_len:
                full_seq = full_seq[-max_len:] 
            
            # Decoder Target (Labels)
            if len(target_seq) > max_len:
                target_seq = target_seq[:max_len]

            input_ids = torch.LongTensor(full_seq)
            labels = torch.LongTensor(target_seq)
            
            # Mask 初始化：全 1
            attention_mask = torch.ones_like(input_ids)
            
            # Padding 处理
            if len(input_ids) < max_len:
                pad_len = max_len - len(input_ids)
                # 1. Input Padding
                input_ids = torch.cat([input_ids, torch.tensor([self.padding_token] * pad_len, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])

            if len(target_seq) < max_len :
                pad_label_len = max_len - len(target_seq)
                labels = torch.cat([labels, torch.tensor([self.ignored_label]*pad_label_len)])

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

        # ==========================================================
        # Valid / Test Split
        # ==========================================================
        else:
            history = item_seq
            target = target_seq


            # 1. Encoder Input 处理
            if len(history) > max_len:
                history = history[-max_len:]
            
            input_ids = torch.LongTensor(history)
            attention_mask = torch.ones_like(input_ids)
            path_labels = torch.LongTensor(target_seq[:-1])

            if len(input_ids) < max_len:
                pad_len = max_len - len(input_ids)      
                # Input Pad
                input_ids = torch.cat([input_ids, torch.tensor([self.padding_token] * pad_len, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])

            if len(target_seq) < self.max_token_seq_len :
                pad_label_len = self.max_token_seq_len - len(target_seq)
                path_labels = torch.cat([path_labels, torch.tensor([self.padding_token]*pad_label_len)])

            # 2. Ground Truth 提取 
            target_np = np.array(target)
            p_indices = np.where(target_np == self.purchase_token)[0]
            
            gt_item = self.ignored_label 
            
            if len(p_indices) > 0:
                buy_idx = p_indices[0]
                item_idx = buy_idx + 1
                if item_idx < len(target):
                    gt_item = target[item_idx]

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.LongTensor([gt_item]),
                "path_labels" : path_labels
                # "target_labels": target_labels 
            }



    @property
    def padding_token(self):
        return 0

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return self.eos_token + 1

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length, including the EOS token.

        Returns:
            int: The maximum token sequence length.
        """
        return self.config['max_item_seq_len'] 
    
    def _tokens2item(self, token) -> str:
        if isinstance(token, list):
            return [self._tokens2item(t) for t in token]
        return self.tokens2item.get(token, "None")
    