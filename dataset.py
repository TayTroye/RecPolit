import os
import glob
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
from logging import getLogger
from utils import load_json
import random
import warnings
import sys


# for tmall dataset

class SeqRecDataset(Dataset):
    def __init__(self, config, split="train", sample_ratio=1.0):
        self.config = config
        self.split = split
        self.sample_ratio = sample_ratio
        self.logger = getLogger()

        # -------------------------------------------------
        # 1. Load ID Mapping
        # -------------------------------------------------
        # id_map_path = os.path.join(config['data_dir'], 'id_mapping.json')
        id_map_path = config["id_map_file"]
        self.id_mapping = load_json(id_map_path)
        self.item_num = len(self.item2id)

        # -------------------------------------------------
        # 2. Special Tokens
        # -------------------------------------------------
        self.pad_token = 0
        self.click_token = self.item_num 
        self.cart_token  = self.item_num + 1
        self.collet_token = self.item_num + 2
        self.purchase_token = self.item_num + 3
        self.bos_token = self.item_num + 4   # Session Start
        self.eos_token = self.item_num + 5   # Session End



        # -------------------------------------------------
        # 3. Load & Build Sessions
        # -------------------------------------------------
        df = self._load_parquet_filter(config["data_dir_file"])
        user_sessions = self._build_user_sessions(df)

        self.user_num = len(user_sessions)

        # inter_num: 所有用户、所有 session 中包含的有效交互(item)总和
        # user_sessions 结构: {uid: [[(act, item), (act, item)], [...]]}
        self.inter_num = 0
        for sessions in user_sessions.values():
            for sess in sessions:
                self.inter_num += len(sess)

        # -------------------------------------------------
        # 4. Build Samples
        # -------------------------------------------------


        self.squeeze_data = config.get("squeeze_data", True)

        if self.squeeze_data :
            self.data_samples = self._build_samples_squeeze(user_sessions)

        else:
            self.data_samples = self._build_samples(user_sessions)
        



        self.logger.info(
            f"[{split}] Dataset Info:\n"
            f"\tUser Num: {self.user_num}\n"
            f"\tItem Num: {self.item_num}\n"
            f"\tInter Num: {self.inter_num}\n"
            f"\tSparsity: {self.sparsity():.4f}\n"
            f"\tSample Size: {len(self.data_samples)}"
        )

    # =========================================================
    # Basic Dataset API
    # =========================================================
    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]

    # =========================================================
    # Data Loading
    # =========================================================


    def sparsity(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.inter_num / self.user_num / self.item_num

    def _load_parquet_filter(self, file_path, sample_fraction=0.01, random_state=42):
        full_df = pd.read_parquet(file_path)
        
        # sample 
        # if sample_fraction < 1.0:
        #     sample_size = max(1, int(len(full_df) * sample_fraction))
        #     sampled_df = full_df.sample(n=sample_size, random_state=random_state, ignore_index=True)
        #     print(f"采样结果: 原始数据 {len(full_df)} 条 -> 采样 {len(sampled_df)} 条 ({sample_fraction*100:.1f}%)")
        #     return sampled_df
        
        return full_df

    # =========================================================
    # Session Construction
    # =========================================================
    def _build_user_sessions(self, df):
        """
        user -> [ session1, session2, ... ]
        session = [(action, raw_item_id), ...]
        """
        user_sessions = defaultdict(list)
        if df.empty:
            return user_sessions

        for _, row in tqdm(df.iterrows(), total=len(df), desc="build sessions"):
            uid = row["user_id"]
            gids = row["gid_seq"]
            items = row["item_seq"]
            actions = row["action_list"]

            cur_gid = None
            cur_session = []

            for gid, item, act in zip(gids, items, actions):
                if str(gid) == "0":
                    continue

                if gid != cur_gid:
                    if cur_session:
                        user_sessions[uid].append(cur_session)
                    cur_session = []
                    cur_gid = gid
                # if item < 0 :
                #     print(f"found negtatibe item id :{item } in uid :{uid}")
                #     exit()

                cur_session.append((str(act), item))

            if cur_session:
                user_sessions[uid].append(cur_session)

        return user_sessions

    # =========================================================
    # Session Type Analysis
    # =========================================================
    def _analyze_session_type(self, session):
        actions = [act for act, _ in session]  # 提取所有动作
        purchase_cnt = sum(act == "alipay" for act in actions)  # 计算 'alipay' 出现的次数

        if purchase_cnt == 0:
            return "no_alipay"

        if purchase_cnt != 1:
            return "invalid"

        purchase_idx = next(
            i for i, act in enumerate(actions) if act == "alipay"
        )

        has_click_before = any(
            actions[i] != "alipay" for i in range(purchase_idx)
        )

        unique_items = {
            session[i][1] for i in range(purchase_idx + 1)
        }

        if not has_click_before or len(unique_items) < 2:
            return "invalid"

        return "valid_purchase"


    # =========================================================
    # Sample Construction
    # =========================================================

    def _build_samples_squeeze(self, user_sessions):
        samples = []

        for user, raw_sessions in tqdm(user_sessions.items(), desc="build samples"):
            
            # ------------------------------------------------------
            # 1. 数据清洗 (保持不变)
            # ------------------------------------------------------
            clean_sessions = []
            purchase_indices = [] 

            for sess in raw_sessions:
                sType = self._analyze_session_type(sess)
                # 只保留纯点击和有效购买
                if sType in ["no_alipay", "valid_purchase"]:
                    clean_sessions.append(sess)
                    if sType == "valid_purchase":
                        purchase_indices.append(len(clean_sessions) - 1)

            if not clean_sessions:
                continue

            n = len(purchase_indices)
            p = purchase_indices
            sessions = clean_sessions
            
            # ------------------------------------------------------
            # 2. 确定划分边界 (Train / Valid / Test)
            # ------------------------------------------------------
            # 我们先确定 Valid 和 Test 的 Session 在列表中的下标
            # 这样 Train 的范围就是 [0, valid_idx - 1]
            
            valid_idx = -1
            test_idx = -1


            if n <= 2:
                # 这些数据只用于训练集
                if self.split != "train": 
                    continue
                
                # 场景 1: n=0 (无购买) -> 丢弃
                if n == 0:
                    continue
                
                # 场景 2: n=1 (单次购买)
                if n == 1:
                    # 如果这唯一的购买发生在第 0 个 session，意味着没有 History，无法预测 -> 丢弃
                    if p[0] == 0:
                        continue
                    # 否则 (购买在后面，有点击流做历史) -> 全量训练
                    else:
                        valid_idx = len(sessions)

                # 场景 3: n=2 (两次购买) -> 全量训练
                # 即便 p[0]==0 (第一次购买在开头)，循环从 i=1 开始也会自动跳过它，
                # 而去尝试预测 p[1] (第二次购买)
                if n == 2:
                    valid_idx = len(sessions)

                
            # Case B: 购买足够多，进行正常划分
            else:
                if p[0] != 0:
                    test_idx = p[-1]
                    valid_idx = p[-2]
                else:
                    valid_idx = p[-1]
                    test_idx = -1

            purchase_set = set(purchase_indices)

            # ======================================================
            # 3. 核心逻辑：Train Split 的滑动窗口生成
            # ======================================================
            if self.split == "train":
                # 训练范围：从第 2 个 session 开始，一直到 Valid Session 之前
                # 能够作为 Target 的 session 下标范围: [1, valid_idx - 1]
                # 对应的 Input 范围: [0] ... [0, ..., i-1]
                
                # 如果 valid_idx 为 -1 (异常) 或 0 (没有历史)，则无法构建训练对
                end_range = valid_idx if valid_idx != -1 else 0
                
                for i in range(1, end_range):
                    # --- Input (Context) ---
                    # 历史必须是完整的 Session 序列
                    # 例如预测 S2 时，Input 是 S1(全量)
                    # 预测 S3 时，Input 是 S1(全量) + S2(全量)
                    if i not in purchase_set:
                        continue

                    input_seq = []
                    for sess in sessions[:i]:
                        input_seq.extend(self._flatten_session_squeeze(sess))
                    
                    # --- Target (Label) ---
                    # 当前要预测的 Session
                    target_sess = sessions[i]
                    raw_target_seq = self._flatten_session_squeeze(target_sess)
                    
                    # ★关键逻辑：Label 截断★
                    # 找到购买行为，截取到 [购买Action, 购买Item] 为止
                    final_target_seq = self._truncate_target_at_purchase(raw_target_seq)
                    
                    # 只有当 target 有内容时才加入样本
                    # if n == 1:
                    #     print("item_seq",input_seq)
                    #     print("target_Seq",final_target_seq)
                    if input_seq and final_target_seq:

                        if n == 1:
                            # 检查 purchase_token 是否在序列中
                            has_token = self.purchase_token in final_target_seq
                            
                            if not has_token :
                                print("    ❌ Check Failed: MISSING purchase token!")
                                raise ValueError("Stop for debugging")
                        

                        samples.append({
                            "user": user,
                            "item_seq": input_seq,      # 完整的历史
                            "target_seq": final_target_seq # 截断后的当前目标
                        })

            # ======================================================
            # 4. Valid Split (通常只预测一个特定的 Session)
            # ======================================================
            elif self.split == "valid":
                if valid_idx != -1:
                    # Input: Valid 之前的所有历史 (全量)
                    prefix = []
                    for sess in sessions[:valid_idx]:
                        prefix.extend(self._flatten_session_squeeze(sess))
                    
                    # Target: Valid Session 本身 (截断)
                    target_sess = sessions[valid_idx]
                    raw_target_seq = self._flatten_session_squeeze(target_sess)
                    final_target = self._truncate_target_at_purchase(raw_target_seq)

                    samples.append({
                        "user": user,
                        "item_seq": prefix,
                        "target_seq": final_target
                    })

            # ======================================================
            # 5. Test Split
            # ======================================================
            elif self.split == "test":
                if test_idx != -1:
                    prefix = []
                    for sess in sessions[:test_idx]:
                        prefix.extend(self._flatten_session_squeeze(sess))
                    
                    target_sess = sessions[test_idx]
                    raw_target_seq = self._flatten_session_squeeze(target_sess)
                    final_target = self._truncate_target_at_purchase(raw_target_seq)

                    samples.append({
                        "user": user,
                        "item_seq": prefix,
                        "target_seq": final_target
                    })

        return samples


    def _build_samples(self, user_sessions):
        samples = []

        for user, raw_sessions in tqdm(user_sessions.items(), desc="build samples"):
            
            # ------------------------------------------------------
            # 1. 数据清洗 (保持不变)
            # ------------------------------------------------------
            clean_sessions = []
            purchase_indices = [] 

            for sess in raw_sessions:
                sType = self._analyze_session_type(sess)
                # 只保留纯点击和有效购买
                if sType in ["no_alipay", "valid_purchase"]:
                    clean_sessions.append(sess)
                    if sType == "valid_purchase":
                        purchase_indices.append(len(clean_sessions) - 1)

            if not clean_sessions:
                continue

            n = len(purchase_indices)
            p = purchase_indices
            sessions = clean_sessions
            
            # ------------------------------------------------------
            # 2. 确定划分边界 (Train / Valid / Test)
            # ------------------------------------------------------
            # 我们先确定 Valid 和 Test 的 Session 在列表中的下标
            # 这样 Train 的范围就是 [0, valid_idx - 1]
            
            valid_idx = -1
            test_idx = -1


            if n <= 2:
                # 这些数据只用于训练集
                if self.split != "train": 
                    continue
                
                # 场景 1: n=0 (无购买) -> 丢弃
                if n == 0:
                    continue
                
                # 场景 2: n=1 (单次购买)
                if n == 1:
                    # 如果这唯一的购买发生在第 0 个 session，意味着没有 History，无法预测 -> 丢弃
                    if p[0] == 0:
                        continue
                    # 否则 (购买在后面，有点击流做历史) -> 全量训练
                    else:
                        valid_idx = len(sessions)

                # 场景 3: n=2 (两次购买) -> 全量训练
                # 即便 p[0]==0 (第一次购买在开头)，循环从 i=1 开始也会自动跳过它，
                # 而去尝试预测 p[1] (第二次购买)
                if n == 2:
                    valid_idx = len(sessions)

                
            # Case B: 购买足够多，进行正常划分
            else:
                if p[0] != 0:
                    test_idx = p[-1]
                    valid_idx = p[-2]
                else:
                    valid_idx = p[-1]
                    test_idx = -1

            purchase_set = set(purchase_indices)

            # ======================================================
            # 3. 核心逻辑：Train Split 的滑动窗口生成
            # ======================================================
            if self.split == "train":
                # 训练范围：从第 2 个 session 开始，一直到 Valid Session 之前
                # 能够作为 Target 的 session 下标范围: [1, valid_idx - 1]
                # 对应的 Input 范围: [0] ... [0, ..., i-1]
                
                # 如果 valid_idx 为 -1 (异常) 或 0 (没有历史)，则无法构建训练对
                end_range = valid_idx if valid_idx != -1 else 0
                
                for i in range(1, end_range):
                    # --- Input (Context) ---
                    # 历史必须是完整的 Session 序列
                    # 例如预测 S2 时，Input 是 S1(全量)
                    # 预测 S3 时，Input 是 S1(全量) + S2(全量)
                    if i not in purchase_set:
                        continue

                    input_seq = []
                    for sess in sessions[:i]:
                        input_seq.extend(self._flatten_session(sess))
                    
                    # --- Target (Label) ---
                    # 当前要预测的 Session
                    target_sess = sessions[i]
                    raw_target_seq = self._flatten_session(target_sess)
                    
                    # ★关键逻辑：Label 截断★
                    # 找到购买行为，截取到 [购买Action, 购买Item] 为止
                    final_target_seq = self._truncate_target_at_purchase(raw_target_seq)
                    
                    # 只有当 target 有内容时才加入样本
                    # if n == 1:
                    #     print("item_seq",input_seq)
                    #     print("target_Seq",final_target_seq)
                    if input_seq and final_target_seq:

                        if n == 1:
                            # 检查 purchase_token 是否在序列中
                            has_token = self.purchase_token in final_target_seq
                            
                            if not has_token :
                                print("    ❌ Check Failed: MISSING purchase token!")
                                raise ValueError("Stop for debugging")
                        

                        samples.append({
                            "user": user,
                            "item_seq": input_seq,      # 完整的历史
                            "target_seq": final_target_seq # 截断后的当前目标
                        })

            # ======================================================
            # 4. Valid Split (通常只预测一个特定的 Session)
            # ======================================================
            elif self.split == "valid":
                if valid_idx != -1:
                    # Input: Valid 之前的所有历史 (全量)
                    prefix = []
                    for sess in sessions[:valid_idx]:
                        prefix.extend(self._flatten_session(sess))
                    
                    # Target: Valid Session 本身 (截断)
                    target_sess = sessions[valid_idx]
                    raw_target_seq = self._flatten_session(target_sess)
                    final_target = self._truncate_target_at_purchase(raw_target_seq)

                    samples.append({
                        "user": user,
                        "item_seq": prefix,
                        "target_seq": final_target
                    })

            # ======================================================
            # 5. Test Split
            # ======================================================
            elif self.split == "test":
                if test_idx != -1:
                    prefix = []
                    for sess in sessions[:test_idx]:
                        prefix.extend(self._flatten_session(sess))
                    
                    target_sess = sessions[test_idx]
                    raw_target_seq = self._flatten_session(target_sess)
                    final_target = self._truncate_target_at_purchase(raw_target_seq)

                    samples.append({
                        "user": user,
                        "item_seq": prefix,
                        "target_seq": final_target
                    })

        return samples

    def _truncate_target_at_purchase(self, seq):
        """
        辅助函数：截断 Target 序列。
        逻辑：遍历序列，一旦遇到 Action Token 为 3 (加购) 或 5 (购买)，
        保留该 Action 和紧随其后的 Item，然后丢弃后面的内容。
        """
        
        truncated_seq = []
        iterator = iter(seq)
        
        try:
            while True:
                token = next(iterator)
                truncated_seq.append(token)
                

                if self._is_purchase_action(token): 
                    next_item = next(iterator)
                    truncated_seq.append(next_item)
                    truncated_seq.append(self.eos_token)
                    break 
                    
        except StopIteration:
            # 如果遍历完都没发现购买行为（可能是纯点击 Session），则保留全部
            pass
            
        return truncated_seq

    def _is_purchase_action(self, token):
        return token == self.purchase_token
     

    def _map_action(self, action):
        if action == "click":
            return self.click_token
        elif action == "alipay":
            return self.purchase_token
        elif action == "cart":
            return self.cart_token
        elif action == "collect":  
            return self.collet_token
        else:
            warnings.warn(f"Warning: Unexpected action '{action}' encountered.")
            print(f"{self.click_token},{self.bos_token},{self.eos_token},action is {action}")
            sys.exit("Stopping execution due to invalid action.")

    # =========================================================
    # Flatten Helpers
    # =========================================================
    def _flatten_session_squeeze(self, session):
        seq = [self.bos_token]
        last_action = None

        for action, item_id in session:
            if item_id is None:
                continue
            
            if action != last_action:
                seq.append(self._map_action(action))
                last_action = action  # 更新状态

            seq.append(self.item2id[str(item_id)])

        seq.append(self.eos_token)

        return seq

    def _flatten_session(self, session):
        seq = [self.bos_token]

        for action, item_id in session:
            if item_id is None:
                continue
            seq.extend([
                self._map_action(action),
                self.item2id[str(item_id)]
            ])

        seq.append(self.eos_token)

        return seq
    
    @property
    def item2id(self):
        return self.id_mapping["item2id"]
