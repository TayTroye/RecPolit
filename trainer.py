import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
from logging import getLogger
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator, PartialState
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from model import AbstractModel
from tokenizer import AbstractTokenizer
from evaluator import Evaluator
from utils import *
import copy
import time
import torch.distributed as dist
import math
import torch.nn.functional as F


class Trainer:

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer, train_dataloader: DataLoader):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.logger = getLogger()
        self.evaluator = Evaluator(config, tokenizer)
        self.tokenizer = tokenizer

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)

        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_n_steps,
        )

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )


        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            f'{self.config["ckpt_name"]}.pth'
        )

        self.results_dir = self.config['results_dir'] if self.config['results_dir'] else self.config['ckpt_dir']
        ensure_dir(self.results_dir)
        
        self.best_epoch = 0
        self.best_val_score = -1
        self.val_delay = self.config['val_delay']
        self.train_batch_size = self.config["train_batch_size"]

        self.state = PartialState()
        self.world_size = self.state.num_processes

        self.device = self.state.device
        
        self.model.device = self.device

        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)


    def save_states(self, epoch=0, path=None):
        path = path if path is not None else self.saved_model_ckpt
        if self.accelerator.is_main_process:
            if self.config['use_ddp']:  # unwrap model for saving
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_optimizer = self.optimizer
                unwrapped_scheduler = self.scheduler
                states = {
                    'model': unwrapped_model.state_dict(),
                    'optimizer': unwrapped_optimizer.state_dict(),
                    'scheduler': unwrapped_scheduler.state_dict()
                }
                torch.save(states, path)
            else:
                states = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                }
                torch.save(states, path)
            self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {path}')

    def load_states(self, ckpt_path=None):
        ckpt_path = self.saved_model_ckpt if ckpt_path is None else ckpt_path
        ckpt = torch.load(ckpt_path, map_location=self.model.device)
        self.log(f'Loading model checkpoint from {ckpt_path}')
        if self.config['use_ddp']:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_optimizer = self.optimizer
            # unwrapped_optimizer = self.accelerator.unwrap_optimizer(self.optimizer)
            unwrapped_scheduler = self.scheduler 
            unwrapped_model.load_state_dict(ckpt['model'])
            unwrapped_optimizer.load_state_dict(ckpt['optimizer'])
            unwrapped_scheduler.load_state_dict(ckpt['scheduler'])
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(unwrapped_model, unwrapped_optimizer, unwrapped_scheduler)
        else:
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])


    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss_dict):
        train_loss_output = (
            "[Epoch %d] [time: %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        if isinstance(loss_dict, dict):
            train_loss_output += "train loss" + str(list(loss_dict.items()))
        else:
            train_loss_output += "train loss" + ": %.4f" % loss_dict
        return train_loss_output + "]"

    def fit(self, train_dataloader, val_dataloader, epochs, epoch_bias=0):

        # Prepare dataloaders
        train_dataloader, val_dataloader = self.accelerator.prepare(
            train_dataloader, val_dataloader
        )

        # 初始化日志追踪器
        self.accelerator.init_trackers(
            project_name=get_file_name(self.config, suffix=''),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {"flush_secs": 60}},
        )

        early_stopping = False

        for epoch in range(epoch_bias, epochs + epoch_bias):
            # -------------------------------
            # Training
            # -------------------------------
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
                disable=not self.accelerator.is_main_process,
            )

            for batch in train_progress_bar:

                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                self.scheduler.step()
                
                train_progress_bar.set_postfix(lr=self.scheduler.get_last_lr(), loss=loss.item())
                total_loss = total_loss + loss.item()


            # log train loss
            self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader)}')

            # -------------------------------
            # Save checkpoints
            # -------------------------------
            if self.config.get('save_interval') is not None and (epoch + 1) % self.config['save_interval'] == 0:
                epoch_ckpt_path = os.path.join(
                    self.config['ckpt_dir'],
                    f'{self.config["ckpt_name"]}_{epoch + 1}.pth'
                )
                self.save_states(epoch=epoch, path=epoch_ckpt_path)

            # -------------------------------
            # Evaluation
            # -------------------------------
            if (epoch + 1) > self.val_delay and (epoch + 1) % self.config['eval_interval'] == 0:
                val_results, _ = self.evaluate(val_dataloader, split='val')
                self.log_results(val_results, epoch, prefix='Val')

                val_score = val_results[self.config['val_metric']]
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.best_epoch = epoch + 1
                    self.save_states(epoch=epoch)

                if self.config.get('patience') is not None and epoch + 1 - self.best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    early_stopping = True
                    break

            self.accelerator.wait_for_everyone()

        self.log(f'Best epoch: {self.best_epoch}, Best val score: {self.best_val_score}')
        if self.best_val_score == -1:
            self.save_states(epoch=epochs + epoch_bias)

        return early_stopping
    
    def evaluate(self, dataloader, split='test', store=False):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
            disable=not self.accelerator.is_main_process,
        )
        all_results_info = {"preds": [], "scores": [], "labels": []}
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']:  # ddp, gather data from all devices for evaluation
                    preds, scores = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_scores, all_labels = self.accelerator.gather_for_metrics(
                        (preds, scores, batch['labels']))
                     
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)


                    all_results_info["preds"].append(all_preds.detach().cpu())
                    all_results_info["scores"].append(all_scores.detach().cpu())
                    all_results_info["labels"].append(all_labels.detach().cpu())
                else:
                    preds, scores = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])
                    all_results_info["preds"].append(preds.detach().cpu())
                    all_results_info["scores"].append(scores.detach().cpu())
                    all_results_info["labels"].append(batch['labels'].detach().cpu())

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()

        for key in all_results_info:
            all_results_info[key] = torch.cat(all_results_info[key], dim=0).tolist()

        if store:
            self.store_results(all_results_info, dataloader.collate_fn, split)

        return output_results, all_results_info


    def store_results(self, results_info, collate_fn, split='test'):
        """
        Store the results in a file.

        Args:
            results_info (dict): The results info to store.
            collate_fn (Collator): The collate function used for data loading.
        """
        preds = results_info['preds']
        pred_ids = []
        for i in range(len(preds)):
            item_list = []
            for j in range(len(preds[i])):
                item = collate_fn.tokens2item(preds[i][j])
                item_list.append(item)
            pred_ids.append(item_list)
        results_info['pred_ids'] = pred_ids

        labels = results_info['labels']
        label_ids = []

        eos_token = collate_fn.tokenizer.eos_token
        # eos_token = collate_fn.tokenizers[0].eos_token
        for i in range(len(labels)):
            cur_label = labels[i]
            if eos_token in cur_label:
                eos_pos = cur_label.index(eos_token)
                cur_label = cur_label[:eos_pos]

            target_item = collate_fn.tokens2item(cur_label)
            label_ids.append(target_item)
        results_info['label_ids'] = label_ids

        if self.accelerator.is_main_process:

            results_info_path = os.path.join(self.results_dir, f"{split}_results.json")

            with open(results_info_path, 'w') as f:
                json.dump(results_info, f)
            self.log(f'Stored results to {results_info_path}')


    def log_results(self, results, epoch, prefix='Val'):

        if self.accelerator.is_main_process:
            for key in results:
                self.accelerator.log({f"{prefix}_Metric/{key}": results[key]}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] {prefix} Results: {results}')
    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
    


class RLTrainer(Trainer):
    def __init__(self, config, model, tokenizer, train_dataloader=None,
                ):
        super(RLTrainer, self).__init__(config, model, tokenizer,
                                        train_dataloader)

        self.beta = config['beta']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_low = 1 - self.epsilon
        self.epsilon_high = 1 + self.epsilon
        self.n_sample = config['n_sample']
        self.group_num = config['group_num']
        self.num_iterations = config['num_iterations']
        # self.reward_type = config['reward_type']
        self.do_sample = config["do_sample"]
        self.group_norm = config['group_norm']
        self.batch_norm = config["batch_norm"]
        self.alpha = config['alpha']
        self.pred_alpha = config["pred_alpha"]
        self.path_beta = config["path_beta"]
        self.len_gamma = config["len_gamma"] 
        self.temperature = config["temperature"]
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.sft_embedding_table = config.get("sft_emb_table_path", False)
        if self.sft_embedding_table:
            self.sft_embedding_table_weight = torch.load(self.sft_embedding_table)
            self.sft_embedding_table_weight.requires_grad_(False)
            self.sft_embedding_table_weight = self.sft_embedding_table_weight.to(model.t5.device)

            self.ignore_ids_list = [
                self.tokenizer.padding_token, # 
                self.tokenizer.click_token, 
                self.tokenizer.collet_token, 
                self.tokenizer.cart_token,
                self.tokenizer.purchase_token,
                self.tokenizer.bos_token,
                self.tokenizer.eos_token,
            ]
            
            self.ignore_ids_tensor = torch.tensor(self.ignore_ids_list, device=model.t5.device).long()



        self.squeeze_data = config.get("squeeze_data", True)

        self.ref_model = None
        if self.beta > 0:
            self.ref_model = copy.deepcopy(model)
            self.ref_model.requires_grad_(False)
            self.ref_model = self.accelerator.prepare(self.ref_model)
            self.ref_model.eval()


    def fit(self, train_dataloader, val_dataloader, epochs, epoch_bias=0):
        train_dataloader, val_dataloader = self.accelerator.prepare(
            train_dataloader, val_dataloader
        )

        # 初始化日志追踪器
        self.accelerator.init_trackers(
            project_name=get_file_name(self.config, suffix=''),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {"flush_secs": 60}},
        )

        early_stopping = False
        for epoch in range(epoch_bias, epochs + epoch_bias):   
            self.accelerator.wait_for_everyone()
            # train
            training_start_time = time.time()
            # print(f"Type of train_dataloader: {type(train_dataloader)}") #<class 'accelerate.data_loader.DataLoaderShard'>
            train_loss = self._train_epoch(train_dataloader,epoch)
            training_end_time = time.time()

            train_loss_output = self._generate_train_loss_output(
                epoch, training_start_time, training_end_time, train_loss
            )

            self.log(train_loss_output+f' LR: {round(self.scheduler.get_last_lr()[0], 7)}')
            

            # -------------------------------
            # Save checkpoints
            # -------------------------------
            if self.config.get('save_interval') is not None and (epoch + 1) % self.config['save_interval'] == 0:
                epoch_ckpt_path = os.path.join(
                    self.config['ckpt_dir'],
                    f'{self.config["ckpt_name"]}_{epoch + 1}.pth'
                )
                self.save_states(epoch=epoch, path=epoch_ckpt_path)


            # -------------------------------
            # Evaluation
            # -------------------------------
            if (epoch + 1) > self.val_delay and (epoch + 1) % self.config['eval_interval'] == 0:
                val_results, _ = self.evaluate(val_dataloader, split='val')
                self.log_results(val_results, epoch, prefix='Val')

                val_score = val_results[self.config['val_metric']]
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.best_epoch = epoch + 1
                    self.save_states(epoch=epoch)

                if self.config.get('patience') is not None and epoch + 1 - self.best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    early_stopping = True
                    break

            self.accelerator.wait_for_everyone()

        self.log(f'Best epoch: {self.best_epoch}, Best val score: {self.best_val_score}')
        if self.best_val_score == -1:
            self.save_states(epoch=epochs + epoch_bias)

        return early_stopping

    def _train_epoch(self, train_dataloader, epoch_idx, verbose=True):

        self.model.train()

        total_num = 0
        total_loss = 0
        total_reward = 0
        train_dataloader = train_dataloader
        # print(f"Type of train_dataloader: {type(train_dataloader)}") #int
        iter_data = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch_idx + 1}]",
                disable=not self.accelerator.is_main_process,
            )
        
        total_reward_info = {
                    "raw_item": 0.0, 
                    "raw_path": 0.0, 
                    "raw_len": 0.0
                }

        for batch_idx, batch in enumerate(iter_data):
            with self.accelerator.accumulate(self.model):
                with torch.no_grad():
                    inputs = self._prepare_inputs(batch)

                for _ in range(self.num_iterations):
                    total_num += 1
                    self.optimizer.zero_grad()

                    loss, reward, batch_re_info = self.compute_loss(inputs, epoch_idx)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                    self.scheduler.step()
                    loss = self.accelerator.gather(loss).mean().item()
                    reward_list = self.accelerator.gather(reward).view(self.world_size, -1).mean(dim=0)
                    reward = reward_list.mean().item()


                    for k, v in batch_re_info.items():
                        if k in total_reward_info:
                            avg_v = self.accelerator.gather(v).mean().item()
                            total_reward_info[k] += avg_v

                    total_loss += loss
                    total_reward += reward_list
                    iter_data.set_postfix(loss=loss, reward=reward, 
                                          item_acc=batch_re_info['raw_item'].item(),path_reward=batch_re_info['raw_path'].item(),
                                          process_reward=batch_re_info['raw_process_token'].item(),action_reward =batch_re_info["raw_action"].item(),
                                          len_acc=batch_re_info['raw_len'].item())

        self.accelerator.wait_for_everyone()

        return dict(loss=round(total_loss/total_num, 5),
                    reward=(total_reward/total_num).detach().cpu())
    
    def _split_completion(self, completion_ids, completion_mask):
        """
        Args:
            completion_ids: [B*N, Seq_Len]
            completion_mask: [B*N, Seq_Len] (1 for valid, 0 for pad)
        Returns:
            item_ids: [B*N, 1] 
            path_preds: [B*N, Seq_Len] 
        """
        pad_id = self.tokenizer.padding_token
        device = completion_ids.device
        
       
        valid_lens = completion_mask.sum(dim=-1).long() # [B*N]
        
        item_indices = (valid_lens - 1).clamp(min=0).unsqueeze(-1) # [B*N, 1]
        item_ids = torch.gather(completion_ids, 1, item_indices)
        
        
        path_preds = completion_ids.clone()
        row_indices = torch.arange(path_preds.size(0), device=device)
        
        path_preds[row_indices, item_indices.squeeze(-1)] = pad_id
        
        return item_ids, path_preds

    def _calculate_action_reward_squeeze(self, path_preds, valid_lens):

        batch_size, seq_len = path_preds.shape
        scores = torch.zeros(batch_size, device=path_preds.device)

        if seq_len == 0:
            return scores

        action_ids = self.action_ids
        
        action_mask = torch.zeros_like(path_preds, dtype=torch.bool)
        for aid in action_ids:
            action_mask |= (path_preds == aid)
                    
        if seq_len > 1:
            curr_is_action = action_mask[:, :-1] # t
            next_is_action = action_mask[:, 1:]  # t+1
            
            consecutive_actions = curr_is_action & next_is_action
            consecutive_counts = consecutive_actions.sum(dim=1).float()
            scores -= consecutive_counts * 0.3 

        # =======================================================
        #  (BOS/EOS)
        # =======================================================
        has_bos = (path_preds == self.tokenizer.bos_token).any(dim=1)
        scores -= has_bos.float() * 0.2
        has_eos = (path_preds == self.tokenizer.eos_token).any(dim=1)
        scores -= has_eos.float() * 0.2 

        first_is_purchase = (path_preds[:, 0] == self.tokenizer.purchase_token)
        scores -= first_is_purchase.float() * 0.3

        return scores

    def sft_emb_reward(self, path_preds, path_labels):


        """
        Batch Process
        
        Args:
            path_preds: [Batch, Pred_Len] (生成的 item ids)
            path_labels: [Batch, Label_Len] (真实的 action/item 混合序列)
            
        Returns:
            rewards
        """            
        # [Batch, Pred_Len, Hidden]
        pred_embs = F.embedding(path_preds, self.sft_embedding_table_weight)
        # [Batch, Label_Len, Hidden]
        label_embs = F.embedding(path_labels, self.sft_embedding_table_weight)
        
        # 3. L2 归一化
        pred_embs = F.normalize(pred_embs, p=2, dim=-1)
        label_embs = F.normalize(label_embs, p=2, dim=-1)
        
        sim_matrix = torch.bmm(pred_embs, label_embs.transpose(1, 2))
        target_mask = torch.isin(path_labels, self.ignore_ids_tensor)
        target_mask = target_mask.unsqueeze(1)
        sim_matrix = sim_matrix.masked_fill(target_mask, -1e9)

        max_sim_scores, _ = sim_matrix.max(dim=-1)

        is_ignore_pred = torch.isin(path_preds, self.ignore_ids_tensor)        
        valid_pred_mask = (~is_ignore_pred).float()
        sum_scores = (max_sim_scores * valid_pred_mask).sum(dim=-1)        
        valid_lens = valid_pred_mask.sum(dim=-1) + 1e-8 # 避免除零
        

        rewards = sum_scores / valid_lens
        
        return rewards


    def reward_function(self, completion_ids,completion_mask, path_labels,item_labels):
        """
            completion_ids: [batch_size*group_num, seq_len]
            preds_labels: [batch_size*group_num, 1]
            path_labels : [batch_size*group_Num,seq_len ]

        """
        item_preds, path_preds = self._split_completion(completion_ids,completion_mask)
        content_seq = path_preds[:, 2:]
        content_mask = (content_seq != self.tokenizer.padding_token)
        content_len = content_mask.sum(dim=1)

        min_len = min(path_preds.shape[1], path_labels.shape[1])  
    
        p_preds = path_preds[:, 2:min_len] 
        p_labels = path_labels[:, 1:min_len-1] 


        # 全匹配 (B, 1) -> (B,)
        item_match = (item_preds == item_labels).all(dim=-1).float()
        item_reward = item_match * 2.0

        path_gt_mask = path_labels != 0 
        path_gt_len = path_gt_mask.sum(dim=1)

        path_pred_mask = path_preds != 0  #
        path_pred_len = path_pred_mask.sum(dim=1) 

        len_diff = torch.abs(path_pred_len - path_gt_len).float() #
        len_rewards = torch.exp(-len_diff / 5.0)  

        token_match_reward = self.sft_emb_reward(path_preds[:, 2:] ,path_labels[:, 1:])
        action_reward = self._calculate_action_reward_squeeze(content_seq, content_len)

        path_rewards = token_match_reward + action_reward  
    

        rewards = self.pred_alpha * item_reward + self.path_beta * path_rewards + self.len_gamma * len_rewards
        reward_info = {
                    "raw_item": item_reward.detach().mean(),       # 
                    "raw_path": path_rewards.detach().mean(),      # 
                    "raw_len": len_rewards.detach().mean(), 
                    "raw_process_token": token_match_reward.detach().mean(), 
                    "raw_action":action_reward.detach().mean(),      #
                    "weighted_total": rewards.detach().mean()      # 加权后的总分
                }

        return rewards,reward_info
    
    def _get_decoder_input(self,completion_ids):

        batch_size, len_labels  = completion_ids.shape
        device = completion_ids.device

        decoder_bos_token =  torch.full(
            (batch_size, 1),
            self.tokenizer.padding_token,
            device=device,
        )

        session_bos_token = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token,
            device=device,
        )

        decoder_input_ids = torch.cat([decoder_bos_token,session_bos_token],dim = -1)

        pading_len = len_labels - decoder_input_ids.shape[1]
        padding_token = torch.full(
            (batch_size, pading_len),
            self.tokenizer.padding_token,
            device=device,
        )
        decoder_input_ids = torch.cat([decoder_input_ids,padding_token],dim = -1)
        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long, device=device)

        return decoder_input_ids, decoder_attention_mask

    def _get_per_token_logps(self, model, input_ids, attention_mask,
                             completion_ids, logits_to_keep, batch_size=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []

        for i in range(0, input_ids.size(0), batch_size): 
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            labels_batch = completion_ids[i : i + batch_size]

            decoder_input_ids, decoder_attention_mask = self._get_decoder_input(completion_ids[i : i+batch_size])

            inputs = {'input_ids': input_ids_batch, 'attention_mask': attention_mask_batch,
                         'labels': labels_batch,"decoder_input_ids": decoder_input_ids,
                            "decoder_attention_mask": decoder_attention_mask}
            logits = model(inputs).logits
            logits = logits[:, -logits_to_keep:]
            labels_batch = labels_batch[:, -logits_to_keep:]
            # logits = logits / self.temperature
            logits = logits / 0.5
            logps = selective_log_softmax(logits, labels_batch) # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _have_path_preds(self, pred_ids, path):
        """
        Args:
            pred_ids: [B, N, 1] 
            path: [B, N, Len]
        Returns:
            completion_ids: [B, N, New_Len]
            completion_mask: [B, N, New_Len]
        """
        batch_size, num_beams, seq_len = path.shape
        pad_id = self.tokenizer.padding_token
        device = path.device
        
        flat_path = path.view(-1, seq_len)
        flat_preds = pred_ids.view(-1)
        
        # valid_lens shape: [B*N]
        valid_lens = (flat_path != pad_id).sum(dim=1)
        
        new_max_len = seq_len + 1
        
        new_path = torch.full(
            (batch_size * num_beams, new_max_len), 
            pad_id, 
            dtype=path.dtype, 
            device=device
        )
        
        new_path[:, :seq_len] = flat_path
        row_indices = torch.arange(flat_path.size(0), device=device)
        new_path[row_indices, valid_lens] = flat_preds
        
    
        new_valid_lens = valid_lens + 1
        
        positions = torch.arange(new_max_len, device=device).unsqueeze(0)
        
        flat_mask = (positions < new_valid_lens.unsqueeze(1)).long()
        
        
        completion_ids = new_path.view(batch_size, num_beams, new_max_len)
        completion_mask = flat_mask.view(batch_size, num_beams, new_max_len)
        
        return completion_ids, completion_mask



    def _split_labels(self, labels):
        """
        Args:
            labels: [B, L] 
        Returns:
            path_labels: [B, Max_Path_Len]  
            pred_labels: [B, 1]    
        """
        batch_size = labels.size(0)
        purchase_id = self.tokenizer.purchase_token
        pad_id = self.tokenizer.padding_token
        device = labels.device

        path_list = []
        pred_list = []

        for i in range(batch_size):
            row = labels[i]
            
            inds = (row == purchase_id).nonzero(as_tuple=True)[0]
            idx = inds[0].item()
            curr_path = row[:idx+1]
            curr_pred = row[idx+1]
            path_list.append(curr_path)
            pred_list.append(curr_pred)

        
        # shape: [B, Max_Len]
        path_labels = torch.nn.utils.rnn.pad_sequence(
            path_list, batch_first=True, padding_value=pad_id
        )

        pred_labels = torch.stack(pred_list).unsqueeze(-1)

        return pred_labels, path_labels

    def _prepare_inputs(self, batch):
        self.model.eval()
        prompt_ids, prompt_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels'] #[B,L]

        preds_labels, path_labels = self._split_labels(labels)

        path_labels = path_labels.unsqueeze(1).repeat(1, self.group_num , 1).view(-1, path_labels.size(-1))
        preds_labels = preds_labels.unsqueeze(1).repeat(1, self.group_num, 1).view(-1, preds_labels.size(-1)) #[B*group_num,1]

        prompt_inputs = {'input_ids': prompt_ids, 'attention_mask': prompt_mask}
        n_return_sequences = self.group_num
        # print(n_return_sequences,self.do_sample)

        if dist.is_initialized():
            preds_items, final_scores, path = self.model.module.generate_rl(prompt_inputs,
                                                        n_return_sequences=n_return_sequences)
            #[B,n_return,1] [B,n_return,Len]
        else:
            preds_items, final_scores, path = self.model.generate_rl(prompt_inputs,
                                                n_return_sequences=n_return_sequences)
        

        completion_ids,completion_mask = self._have_path_preds(preds_items,path) #[B,n,L]

        prompt_ids = prompt_ids.unsqueeze(1).repeat(1, self.group_num, 1).view(-1, prompt_ids.size(-1)) #[B*group_num,L]
        prompt_mask = prompt_mask.unsqueeze(1).repeat(1, self.group_num, 1).view(-1, prompt_mask.size(-1)) 
        completion_ids = completion_ids.view(-1, 1, completion_ids.size(-1))
        completion_ids = completion_ids.squeeze(1)  # [batch_size * group_num, seq_len] 
        completion_mask = completion_mask.view(-1, completion_mask.size(-1))       
        rewards,rewards_info = self.reward_function(completion_ids,completion_mask, path_labels,preds_labels)
        
        logits_to_keep = completion_mask.sum(dim=1)
        batch_size = self.train_batch_size

        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_ids, prompt_mask, completion_ids, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None


        
        if self.group_norm: 
            # 
            if len(rewards.shape) == 1:
                rewards = rewards.view(-1, 1)
            seq_len = rewards.shape[-1]
            
            # [B, G, L] -> 在 G 维度求均值
            mean_grouped_rewards = rewards.view(-1, self.group_num, seq_len).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.group_num, seq_len).std(dim=1)

            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.group_num, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.group_num, dim=0)
            
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            
        else:
            advantages = rewards

        self.model.train()
        
        return {
            "prompt_ids": prompt_ids.contiguous(),
            "prompt_mask": prompt_mask.contiguous(),
            "completion_ids": completion_ids.contiguous(),
            "completion_mask": completion_mask,
            "advantages": advantages,
            "rewards": rewards,
            "rewards_info": rewards_info,
            "old_per_token_logps": old_per_token_logps,
            "batch_input": batch,
        }


    def compute_loss(self, inputs, epoch_idx):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        rewards = inputs["rewards"].mean(dim=0)
        rewards_info = inputs["rewards_info"]
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(self.model, prompt_ids, prompt_mask, completion_ids, logits_to_keep)

        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_ids, prompt_mask, completion_ids, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_ids, prompt_mask, completion_ids, logits_to_keep
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high) #clip grpo loss

        if len(advantages.shape) == 1:
            advantages = advantages.unsqueeze(1)

        zero_mask = (advantages != 0).long()
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl  

        loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()


        return loss, rewards,rewards_info
    
    
    
