import argparse
import os
from logging import getLogger

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import math
import numpy as np
import yaml
import importlib
from accelerate import Accelerator

from collator import Collator
from trainer import Trainer,RLTrainer
from utils import *
from data_utils import *
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TMALL', help='Dataset name: TMALL')
    parser.add_argument('--config_file', type=str, default='./config/sftconfig.yaml', help='Config file')
    return parser.parse_known_args()


def main(config):
    
    init_seed(config['rand_seed'], config['reproducibility'])
    init_logger(config)

    logger = getLogger()
    accelerator = config['accelerator']
    log(f'Device: {config["device"]}', accelerator, logger)
    log(f'Config: {str(config)}', accelerator, logger)

    tokenizer = get_tokenizer(config)
    train_dataset, valid_dataset, test_dataset = get_datasets(config)
    train_collate_fn = Collator(config, tokenizer,split="train")
    test_collate_fn = Collator(config, tokenizer,split="valid")

    model_name = config['model_name']
    model_class = getattr(importlib.import_module(f'models.{model_name}.model'), model_name)

    with accelerator.main_process_first():
        model = model_class(config, train_dataset, tokenizer)
    log(model, accelerator, logger)
    log(model.n_parameters, accelerator, logger)

    train_data = get_dataloader(config, train_dataset, train_collate_fn, 'train')
    valid_data = get_dataloader(config, valid_dataset, test_collate_fn,'valid')
    test_data = get_dataloader(config, test_dataset, test_collate_fn, 'test')

    if config['val_delay'] >= config['epochs']:
        config['val_delay'] = config['epochs'] - 1

    trainer = Trainer(config, model, tokenizer, train_data)
    trainer.fit(train_data, valid_data, config['epochs'])

    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    if config["test_num_beams"] is not None:
        model.config['num_beams'] = config["test_num_beams"]

    model_states = torch.load(trainer.saved_model_ckpt, map_location=trainer.model.device)['model']
    model.load_state_dict(model_states)

    if accelerator.is_main_process:
        log(f'Loaded best model checkpoint from {trainer.saved_model_ckpt}', accelerator, logger)

    trainer.model, test_data = accelerator.prepare(
        model, test_data
    )
    test_results, _ = trainer.evaluate(test_data, split='test', store=True)

    if accelerator.is_main_process:
        for key in test_results:
            accelerator.log({f'Test_Metric/{key}': test_results[key]})


    log(f'Test Results: {test_results}', accelerator, logger)
    trainer.end()
    
    
    
if __name__ == '__main__':    
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    # Config
    config = {}
    config.update(yaml.safe_load(open(args.config_file, 'r')))
    config.update(command_line_configs)

    config['run_local_time'] = get_local_time()

    ckpt_name = get_file_name(config)
    config['ckpt_name'] = ckpt_name
    config['dataset'] = args.dataset
    config['data_dir'] = os.path.join(config['data_dir'], config['dataset'])
    config['ckpt_dir'] = os.path.join(config['ckpt_dir'], config['dataset'], ckpt_name)

    config = convert_config_dict(config)

    config['device'], config['use_ddp'] = init_device()
    config['accelerator'] = Accelerator()
    torch.distributed.barrier(device_ids=[int(os.environ['LOCAL_RANK'])])
    
    main(config)


    

