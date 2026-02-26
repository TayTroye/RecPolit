import copy
import importlib
from torch.utils.data import ConcatDataset, DataLoader

from dataset import SeqRecDataset
from tokenizer import  AbstractTokenizer
import random
import torch
import numpy as np

def get_datasets(config):
    train_dataset = SeqRecDataset(config, split='train',sample_ratio=config['train_ratio'])
    valid_dataset = SeqRecDataset(config, split='valid', sample_ratio=config['val_ratio'])
    test_dataset = SeqRecDataset(config, split='test')

    return train_dataset, valid_dataset, test_dataset



def get_less_datasets(config):
    train_dataset = SeqRecDataset(config, split='train')
    valid_dataset = SeqRecDataset(config, split='valid')

    return train_dataset, valid_dataset


import importlib

def get_tokenizer(config):

    model_name = config['model_name'] 
    

    module_path = f'models.{model_name}.tokenizer'
    class_name = f'{model_name}Tokenizer'

    try:
        module = importlib.import_module(module_path)
        tokenizer_class = getattr(module, class_name)
        # print(f'[TOKENIZER] Loaded tokenizer class {tokenizer_class} from {module_path}.py')
        # exit()
        
    except (ImportError, AttributeError) as e:
        raise ValueError(f'Error loading tokenizer for model "{model_name}". '
                         f'Expected file: {module_path}.py, Class: {class_name}. '
                         f'Error details: {e}')


    return tokenizer_class(config,config['sem_id_epoch'])


# def get_tokenizer(model_name: str):
#     """
#     Retrieves the tokenizer for a given model name.

#     Args:
#         model_name (str): The model name.

#     Returns:
#         AbstractTokenizer: The tokenizer for the given model name.

#     Raises:
#         ValueError: If the tokenizer is not found.
#     """
#     try:
#         tokenizer_class = getattr(
#             importlib.import_module(f'mtgrec.models.{model_name}.tokenizer'),
#             f'{model_name}Tokenizer'
#         )
#     except:
#         raise ValueError(f'Tokenizer for model "{model_name}" not found.')
#     return tokenizer_class


# def get_tokenizers(config):
#     tokenizers = []

#     for sem_id_epoch in config["sem_id_epochs"]:
#         tokenizer = MTGRecTokenizer(config, sem_id_epoch)
#         tokenizers.append(tokenizer)
#     if len(tokenizers) ==0:
#         tokenizers.append(Tokenizer(config))

#     return tokenizers


# 1. 定义 worker 初始化函数
# 这个函数会在每个 worker 启动时运行，确保每个 worker 都有一个确定的、不同的种子
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader(config, dataset, collate_fn, split):
    
    # 2. 创建一个 PyTorch 生成器，并设置固定的种子
    # 从 config 中读取种子，如果没有则默认 42
    seed = config.get('seed', 42) 
    g = torch.Generator()
    g.manual_seed(seed)

    if split == 'train':
        dataloader = DataLoader(
            dataset, 
            batch_size=config['train_batch_size'], 
            collate_fn=collate_fn,
            num_workers=config['num_proc'], 
            shuffle=True,
            # 3. 关键修改：传入 init_fn 和 generator
            worker_init_fn=seed_worker,
            generator=g 
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=config['eval_batch_size'], 
            collate_fn=collate_fn,
            num_workers=config['num_proc'], 
            shuffle=False,
            # 验证集虽然不shuffle，但如果有随机逻辑（如代码中有遗漏的随机操作），加上也是好的习惯
            worker_init_fn=seed_worker,
            generator=g
        )

    return dataloader

def get_dataloader_base(config, dataset, collate_fn, split):

    if split == 'train':
        dataloader = DataLoader(dataset, batch_size=config['train_batch_size'] , collate_fn=collate_fn,
                                num_workers=config['num_proc'], shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=config['eval_batch_size'], collate_fn=collate_fn,
                                num_workers=config['num_proc'], shuffle=False)


    return dataloader
