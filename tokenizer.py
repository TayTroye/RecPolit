import os
from logging import getLogger

import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
from utils import *


class AbstractTokenizer:

	def __init__(self, config):

		self.config = config
		self.logger = getLogger()
		self.eos_token = None



	@property
	def vocab_size(self):
		raise NotImplementedError('Vocabulary size not implemented.')

	def tokenize(self, example:dict) -> dict:
		raise NotImplementedError("tokenization not implemented")



	@property
	def max_token_seq_len(self):
		raise NotImplementedError("Maximum token sequence length not implemented")


	def log(self, message, level='info'):

		return log(message, self.config['accelerator'], self.logger, level=level)





