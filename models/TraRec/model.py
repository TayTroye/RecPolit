import torch
from transformers import T5ForConditionalGeneration, T5Config,LogitsProcessor, LogitsProcessorList

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from model  import AbstractModel
import math
import torch.nn.functional as F
import csv
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn.utils.rnn import pad_sequence
import random

class EntropyStoppingLogitsProcessor(LogitsProcessor):
    def __init__(self, item_num, eos_token_id, bos_token_id,click_token_id, threshold=10.0):
        """
        Args:
            eos_token_id: 停止符 ID (即 purchase_token)
            threshold: 熵值阈值。
        """
        self.eos_token_id = eos_token_id
        self.threshold = threshold

        self.item_num = item_num
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.click_token = click_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [Batch_Size * Gen_K, Vocab_Size] (Logits)

        log_probs = F.log_softmax(scores, dim=-1) # [B, V]
        probs = torch.exp(log_probs)              # [B, V]
        
        # Formula: H(p) = - sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=-1) # [B]

        # if random.random() < 0.001:
        #     print(f"[Entropy Monitor] Mean Entropy: {entropy.mean().item():.4f}")
        
        mask = entropy < self.threshold 
        # print()
        
        # 4. 强制干预
        if mask.any():
            scores[mask, :self.click_token] = -float('inf')
            scores[mask, self.eos_token_id] = -float('inf')
            scores[mask, self.bos_token_id] = -float('inf')

        return scores

class TraRec(AbstractModel):

    def __init__(
        self,
        config,
        dataset,
        tokenizer,
    ):
        super(TraRec, self).__init__(config, dataset, tokenizer)
        self.max_new_token  = config.get("max_new_token",10)
        self.vocab_size = tokenizer.vocab_size
        self.item_num = dataset.item_num
        self.eos_token_id = dataset.eos_token,
        self.bos_token_id = dataset.bos_token,
        self.click_token_id = dataset.click_token
        self.es_threshold = config.get('es_threshold',10)
        self.use_entropy = config.get("use_entropy",True)

        t5config = T5Config(
            num_layers=config['num_layers'], 
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            activation_function=config['activation_function'],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            decoder_start_token_id=0,
            feed_forward_proj=config['feed_forward_proj'],
            n_positions=tokenizer.max_len + 1,
        )

        self.t5 = T5ForConditionalGeneration(config=t5config)

        self.purchase_loss_weight = config["purchase_loss_weight"]

        self.n_neg = config["n_neg"]

        self.log_file = config['log_file']
        self.search_k = config["search_k"]

        self.use_sim = config.get("use_sim", True)

        self.sft_embedding_table = config.get("sft_emb_table_path", False)
        if self.sft_embedding_table:
            self.sft_embedding_table_weight = torch.load(self.sft_embedding_table)
            self.sft_embedding_table_weight.requires_grad_(False)

        # self.sim_threshold = config.get("sim_threshold", 0.85)




    @property
    def n_parameters(self) -> str:
        """
        Calculates the number of trainable parameters in the model.

        Returns:
            str: A string containing the number of embedding parameters, non-embedding parameters, and total trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.t5.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict) -> torch.Tensor:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']     
        neg_ids = batch.get('neg_ids', None)
    
        if neg_ids is not None:     
            decoder_input_ids = self.t5._shift_right(labels)
                    
        
            encoder_outputs = self.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            decoder_outputs = self.t5.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                return_dict=True
            )
            
            #  [Batch, Seq_Len, Hidden_Size]
            hidden_states = decoder_outputs.last_hidden_state 

           
            hidden_states = hidden_states.view(-1, hidden_states.size(-1)) # [N_Total, Hidden]
            labels = labels.view(-1)                                       # [N_Total]
            
            neg_ids = neg_ids.view(-1, self.n_neg) # [N_Total, n_neg]
            
            mask = (labels != self.tokenizer.ignored_label) 
            
            valid_hidden = hidden_states[mask]  # [N_Valid, Hidden]
            valid_labels = labels[mask]         # [N_Valid]
            valid_neg_ids = neg_ids[mask]       # [N_Valid, n_neg]

            # =============================================================
            # 4. 计算 Logits (负采样)
            # =============================================================
            
            embedding_weight = self.t5.shared.weight 
            pos_embeddings = F.embedding(valid_labels, embedding_weight) # [N_Valid, Hidden]
            pos_logits = torch.sum(valid_hidden * pos_embeddings, dim=-1, keepdim=True) # [N_Valid, 1]

            
            neg_embeddings = F.embedding(valid_neg_ids, embedding_weight) # [N_Valid, n_neg, Hidden]
            # [N, n_neg, H] * [N, H, 1] -> [N, n_neg, 1]
            neg_logits = torch.bmm(neg_embeddings, valid_hidden.unsqueeze(2)).squeeze(2) # [N_Valid, n_neg]
            logits = torch.cat([pos_logits, neg_logits], dim=1)
            
    
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=labels.device)
            raw_loss = F.cross_entropy(logits, targets, reduction='none') 
            # 识别是否是 purchase 行为 (加权)
            is_purchase = (valid_labels == self.tokenizer.purchase_token)
            
            sample_weights = torch.ones_like(raw_loss)
            sample_weights[is_purchase] = self.purchase_loss_weight
            
            loss = (raw_loss * sample_weights).mean()

            return Seq2SeqLMOutput(loss=loss, logits=None)
        
        else:
            outputs = self.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=batch.get('decoder_input_ids'),
                decoder_attention_mask=batch.get('decoder_attention_mask'),
                labels=labels #
            )

            return outputs



    def generate_rl(self,batch,n_return_sequences=1): 
        history_ids = batch['input_ids']
        target_labels = batch["pred_label"]
        batch_size = history_ids.shape[0]
        device = history_ids.device
        log_file = self.log_file

        input_ids = history_ids
        attention_mask = batch["attention_mask"]

        target_k = n_return_sequences          
        oversample_factor = 1                 
        gen_k = target_k * oversample_factor 

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

        if self.use_entropy:
            similarity_processor = EntropyStoppingLogitsProcessor(
                item_num = self.item_num,
                eos_token_id=self.eos_token_id,
                bos_token_id=self.bos_token_id,
                click_token_id=self.click_token_id,
                threshold = self.es_threshold,
            )

            logits_processor_list = LogitsProcessorList([similarity_processor])

        
            generated_output = self.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_token,
                decoder_input_ids = decoder_input_ids,
                num_return_sequences=gen_k,
                pad_token_id=self.tokenizer.padding_token,
                eos_token_id=self.tokenizer.purchase_token,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=self.config["temperature_sample"], #1.2
                do_sample=True,   # 启用采样
                top_p=self.config["top_p_sample"] ,   #0.9     # 使用 Top-p 采样
                logits_processor=logits_processor_list
            )
        
        else: 
            generated_output = self.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_token,
                decoder_input_ids = decoder_input_ids,
                num_return_sequences=gen_k,
                pad_token_id=self.tokenizer.padding_token,
                eos_token_id=self.tokenizer.purchase_token,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=self.config["temperature_sample"], #1.2
                do_sample=True,   # 启用采样
                top_p=self.config["top_p_sample"] ,   #0.9     # 使用 Top-p 采样
                # logits_processor=logits_processor_list
            )  

        sequences = generated_output.sequences # [B*K, 1 + New_Tokens]
        decoder_input_len = decoder_input_ids.size(1)  # 2 ([PAD, BOS])

        flat_token_ids = sequences[:, decoder_input_len:]  # [B*K, New_Tokens]

        step_log_probs = []
        for step_i, step_logits in enumerate(generated_output.scores):
            curr_token_id = flat_token_ids[:, step_i].unsqueeze(-1)
            
            step_denominator = torch.logsumexp(step_logits, dim=-1, keepdim=True)
            step_numerator = step_logits.gather(dim=-1, index=curr_token_id)
            curr_log_prob = (step_numerator - step_denominator).squeeze(-1)
            step_log_probs.append(curr_log_prob) #
            
            del step_logits, step_numerator, step_denominator

        transition_scores = torch.stack(step_log_probs, dim=1)  # [B*K,new_token]
        valid_mask = (flat_token_ids != self.tokenizer.padding_token) 
        transition_scores[~valid_mask] = 0.0
        generated_scores = transition_scores.sum(dim=1)
        
        sequences_scores = generated_scores.view(batch_size, gen_k)
        sequences = sequences.view(batch_size, gen_k, -1) #[B,gen_k,newtoken+1]

        del generated_output 
        torch.cuda.empty_cache()

        sorted_scores, sorted_indices = torch.sort(sequences_scores, dim=1, descending=True)
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
        sorted_sequences = torch.gather(sequences, 1, expanded_indices)
        
        sequences = sorted_sequences    #[B,k,len (k+2)]
        sequences_scores = sorted_scores 
        
        flat_paths = sequences.view(-1, sequences.size(-1)) # [B*K, Path_Len]
        batch_size_k = flat_paths.size(0)
        
        purchase_id = self.tokenizer.purchase_token
        pad_id = self.tokenizer.padding_token
        eos_id = self.tokenizer.eos_token
        
        decoder_input_ids_list = []
        purchase_token_indices = [] 
        

        start_token_tensor = torch.tensor([pad_id], device=device)
        purchase_token_tensor = torch.tensor([purchase_id], device=device)
        eos_id = self.tokenizer.eos_token

        purchase_token_indices = [] 

        eos_id = self.tokenizer.eos_token

        for i in range(batch_size_k):
            row = flat_paths[i]
            eos_indices = (row == eos_id).nonzero(as_tuple=True)[0]
            purchase_indices = (row == purchase_id).nonzero(as_tuple=True)[0]
            
            valid_prefix = None
            if len(eos_indices) > 0:
                cutoff_idx = eos_indices[0].item()
                valid_prefix = row[:cutoff_idx]
                
            elif len(purchase_indices) > 0:
                cutoff_idx = purchase_indices[0].item()
                valid_prefix = row[:cutoff_idx]
            else:
                valid_prefix = row

            seq_with_purchase = torch.cat([valid_prefix, purchase_token_tensor])            
            decoder_input_ids_list.append(seq_with_purchase)
            purchase_token_indices.append(len(seq_with_purchase) - 1)

        decoder_input_ids = pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=pad_id)
        gather_indices = torch.tensor(purchase_token_indices, device=device).unsqueeze(1)
         
        encoder_input_ids = input_ids.repeat_interleave(gen_k, dim=0)
        encoder_attention_mask = attention_mask.repeat_interleave(gen_k, dim=0)

        with torch.no_grad():
            outputs = self.t5(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
        
        logits = outputs.logits
        
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, logits.size(-1))
        
        target_logits = torch.gather(logits, 1, gather_indices_expanded).squeeze(1)
        
        next_token_log_probs = torch.log_softmax(target_logits, dim=-1)
        flat_pred_scores, flat_pred_item_ids = torch.topk(next_token_log_probs, k=gen_k, dim=-1)

        item_scores = flat_pred_scores.view(batch_size, gen_k, gen_k)
        item_ids_matrix = flat_pred_item_ids.view(batch_size, gen_k, gen_k)
        
        path_scores_expanded = sequences_scores.view(batch_size, gen_k, 1)
        
        total_scores_matrix = path_scores_expanded + item_scores
        
        flat_total_scores = total_scores_matrix.view(batch_size, -1)
        
        search_k = self.search_k
        # search_k = min(n_return_sequences * n_return_sequences,40)
        raw_scores, raw_indices_flat = torch.topk(flat_total_scores, k=search_k, dim=1)

        raw_path_indices = raw_indices_flat // gen_k 
        raw_item_rank_indices = raw_indices_flat % gen_k 
        
        flat_item_ids_matrix = item_ids_matrix.view(batch_size, -1)
        raw_pred_item_ids = torch.gather(flat_item_ids_matrix, 1, raw_indices_flat)
        
        raw_path_indices_expanded = raw_path_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
        # Gather 出对应的路径
        raw_sequences = torch.gather(sequences, 1, raw_path_indices_expanded) 


        dedup_item_ids_list = []
        dedup_scores_list = []
        dedup_sequences_list = []
        
        for b in range(batch_size):
            seen_items = set()
            b_items = []
            b_scores = []
            b_seqs = []
            
            for i in range(search_k):
                item_id = raw_pred_item_ids[b, i].item()
                
                if item_id not in seen_items:
                    seen_items.add(item_id)
                    b_items.append(item_id)
                    b_scores.append(raw_scores[b, i])
                    b_seqs.append(raw_sequences[b, i])
                    
                    if len(b_items) == gen_k:
                        break
            
            while len(b_items) < gen_k:
                b_items.append(b_items[-1])
                b_scores.append(b_scores[-1])
                b_seqs.append(b_seqs[-1])
            
            dedup_item_ids_list.append(torch.tensor(b_items, device=device))
            dedup_scores_list.append(torch.stack(b_scores))
            dedup_sequences_list.append(torch.stack(b_seqs))

        final_pred_item_ids = torch.stack(dedup_item_ids_list) 
        final_scores = torch.stack(dedup_scores_list)
        final_sequences = torch.stack(dedup_sequences_list)

        completion_ids = torch.cat([final_sequences,final_pred_item_ids.unsqueeze(-1)],dim=-1)

        return final_pred_item_ids.unsqueeze(-1), final_scores, final_sequences

    def generate(self, batch, n_return_sequences=10, log_file="debug.csv"):
        history_ids = batch['input_ids']
        user_ids = batch["user_id"]
        batch_size = history_ids.shape[0]
        device = history_ids.device
        log_file = self.log_file
        input_ids = history_ids
        attention_mask = batch["attention_mask"]

        target_k = n_return_sequences          
        oversample_factor = 1                 
        gen_k = target_k * oversample_factor  

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

        if self.use_entropy:
            similarity_processor = EntropyStoppingLogitsProcessor(
                item_num = self.item_num,
                eos_token_id=self.eos_token_id,
                bos_token_id=self.bos_token_id,
                click_token_id=self.click_token_id,
                threshold = self.es_threshold,
            )

            logits_processor_list = LogitsProcessorList([similarity_processor])

            generated_output = self.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_token,
                decoder_input_ids = decoder_input_ids,
                num_return_sequences=gen_k,
                pad_token_id=self.tokenizer.padding_token,
                eos_token_id=self.tokenizer.purchase_token,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=self.config["temperature_sample"], #1.2
                do_sample=True,   # 启用采样
                top_p=self.config["top_p_sample"] ,   #0.9     # 使用 Top-p 采样
                logits_processor=logits_processor_list
            )
        
        else:
            generated_output = self.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_token,
                decoder_input_ids = decoder_input_ids,
                num_return_sequences=gen_k,
                pad_token_id=self.tokenizer.padding_token,
                eos_token_id=self.tokenizer.purchase_token,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=self.config["temperature_sample"], #1.2
                do_sample=True,   # 启用采样
                top_p=self.config["top_p_sample"] ,   #0.9     # 使用 Top-p 采样
                # logits_processor=logits_processor_list
            ) 

        sequences = generated_output.sequences # [B*K, 1 + New_Tokens]
        decoder_input_len = decoder_input_ids.size(1)  # 2 ([PAD, BOS])
        flat_token_ids = sequences[:, decoder_input_len:]  # [B*K, New_Tokens]

        step_log_probs = []
        

        for step_i, step_logits in enumerate(generated_output.scores):
            curr_token_id = flat_token_ids[:, step_i].unsqueeze(-1)
            
            step_denominator = torch.logsumexp(step_logits, dim=-1, keepdim=True)
            step_numerator = step_logits.gather(dim=-1, index=curr_token_id)
            curr_log_prob = (step_numerator - step_denominator).squeeze(-1)
            step_log_probs.append(curr_log_prob)
            
            del step_logits, step_numerator, step_denominator

        transition_scores = torch.stack(step_log_probs, dim=1)  # [B*K,new_token]

        valid_mask = (flat_token_ids != self.tokenizer.padding_token)
        transition_scores[~valid_mask] = 0.0

        generated_scores = transition_scores.sum(dim=1)

        sequences_scores = generated_scores.view(batch_size, gen_k)
        sequences = sequences.view(batch_size, gen_k, -1) #[B,gen_k,newtoken+1]

        del generated_output 
        torch.cuda.empty_cache()

        sorted_scores, sorted_indices = torch.sort(sequences_scores, dim=1, descending=True)
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
        sorted_sequences = torch.gather(sequences, 1, expanded_indices)
        
        sequences = sorted_sequences   
        sequences_scores = sorted_scores


        
        flat_paths = sequences.view(-1, sequences.size(-1)) # [B*K, Path_Len]
        batch_size_k = flat_paths.size(0)
        
        purchase_id = self.tokenizer.purchase_token
        pad_id = self.tokenizer.padding_token
        eos_id = self.tokenizer.eos_token
        
        decoder_input_ids_list = []
        purchase_token_indices = [] 

        # start_token_tensor = torch.tensor([pad_id], device=device)
        purchase_token_tensor = torch.tensor([purchase_id], device=device)

        for i in range(batch_size_k):
            row = flat_paths[i]
            
            eos_indices = (row == eos_id).nonzero(as_tuple=True)[0]
            purchase_indices = (row == purchase_id).nonzero(as_tuple=True)[0]
            
            valid_prefix = None

            if len(eos_indices) > 0:
                cutoff_idx = eos_indices[0].item()
                valid_prefix = row[:cutoff_idx]
            elif len(purchase_indices) > 0:
                cutoff_idx = purchase_indices[0].item()
                valid_prefix = row[:cutoff_idx]
            else:
                valid_prefix = row
            seq_with_purchase = torch.cat([valid_prefix, purchase_token_tensor])
            decoder_input_ids_list.append(seq_with_purchase)
            purchase_token_indices.append(len(seq_with_purchase) - 1)

        #  Right Padding 
        decoder_input_ids = pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=pad_id)
        gather_indices = torch.tensor(purchase_token_indices, device=device).unsqueeze(1)
        
        encoder_input_ids = input_ids.repeat_interleave(gen_k, dim=0)
        encoder_attention_mask = attention_mask.repeat_interleave(gen_k, dim=0)

        with torch.no_grad():
            outputs = self.t5(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
        
        # outputs.logits Shape: [B*K, Max_Seq_Len, Vocab_Size]
        logits = outputs.logits
        
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, logits.size(-1))
        target_logits = torch.gather(logits, 1, gather_indices_expanded).squeeze(1)
        next_token_log_probs = torch.log_softmax(target_logits, dim=-1)
        flat_pred_scores, flat_pred_item_ids = torch.topk(next_token_log_probs, k=gen_k, dim=-1)

        item_scores = flat_pred_scores.view(batch_size, gen_k, gen_k)
        item_ids_matrix = flat_pred_item_ids.view(batch_size, gen_k, gen_k)
        
        path_scores_expanded = sequences_scores.view(batch_size, gen_k, 1)
        total_scores_matrix = path_scores_expanded + item_scores
        
        # Shape: [B, Path_K * Item_K]
        flat_total_scores = total_scores_matrix.view(batch_size, -1)
        
        search_k = self.search_k
        raw_scores, raw_indices_flat = torch.topk(flat_total_scores, k=search_k, dim=1)

        raw_path_indices = raw_indices_flat // gen_k 
        raw_item_rank_indices = raw_indices_flat % gen_k 
        
        flat_item_ids_matrix = item_ids_matrix.view(batch_size, -1)
        raw_pred_item_ids = torch.gather(flat_item_ids_matrix, 1, raw_indices_flat)
        

        # [B, search_k, Len]
        raw_path_indices_expanded = raw_path_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
        raw_sequences = torch.gather(sequences, 1, raw_path_indices_expanded)

        
        dedup_item_ids_list = []
        dedup_scores_list = []
        dedup_sequences_list = []
        
        for b in range(batch_size):
            seen_items = set()
            b_items = []
            b_scores = []
            b_seqs = []
            
            for i in range(search_k):
                item_id = raw_pred_item_ids[b, i].item()
                
                if item_id not in seen_items:
                    seen_items.add(item_id)
                    b_items.append(item_id)
                    b_scores.append(raw_scores[b, i])
                    b_seqs.append(raw_sequences[b, i])
                    
                    if len(b_items) == gen_k:
                        break
            
            while len(b_items) < gen_k:
                b_items.append(b_items[-1])
                b_scores.append(b_scores[-1])
                b_seqs.append(b_seqs[-1])
            
            dedup_item_ids_list.append(torch.tensor(b_items, device=device))
            dedup_scores_list.append(torch.stack(b_scores))
            dedup_sequences_list.append(torch.stack(b_seqs))

        final_pred_item_ids = torch.stack(dedup_item_ids_list) 
        final_scores = torch.stack(dedup_scores_list)
        final_sequences = torch.stack(dedup_sequences_list)


        return final_pred_item_ids.unsqueeze(-1), final_scores

