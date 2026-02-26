import random
from logging import getLogger

import torch
import copy
from utils import log
import numpy as np


class Collator:

    def __init__(self, config, tokenizer,split):

        self.config = config
        self.logger = getLogger()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.n_neg = config["n_neg"]
        self.split = split

    def __call__(self, batch):
        all_input_ids, all_attention_mask, all_labels = [], [], []
        for example in batch:
            d = self.tokenizer.tokenize(example,self.split)
            all_input_ids.append(d["input_ids"])
            all_attention_mask.append(d["attention_mask"])
            all_labels.append(d["labels"])

        all_input_ids = torch.stack(all_input_ids, dim=0)
        all_attention_mask = torch.stack(all_attention_mask, dim=0)
        all_labels = torch.stack(all_labels, dim=0)

# 构造基础返回字典
        batch_data = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'labels': all_labels
        }

        # -----------------------------------------------------------
        # 2. 新增：只在 Train 模式下生成负样本
        # -----------------------------------------------------------
        if self.split == 'train':
            batch_size, seq_len = all_input_ids.shape
            
            # 生成负样本 [Batch, Seq, n_neg]
            # 这里的 randint 是在 CPU 上运行的，多卡 Dataloader 会由 PyTorch 自动分发 Seed
            neg_ids = torch.randint(
                low=0,
                high=self.vocab_size,
                size=(batch_size, seq_len, self.n_neg),
                dtype=torch.long
            )
            
            batch_data['neg_ids'] = neg_ids

        return batch_data


        # return {
    	# 	'input_ids': all_input_ids,
    	# 	'attention_mask': all_attention_mask,
    	# 	'labels': all_labels
    	# }


    def tokens2item(self, tokens):
        return self.tokenizer._tokens2item(tokens)

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
