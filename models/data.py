import numpy as np
import difflib
from tqdm import tqdm
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, Union
from transformers import PreTrainedModel
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def load_data(source_path, target_path):
    with open(source_path) as f:
        sources = [line.strip() for line in f]
    with open(target_path) as f:
        targets = [line.strip() for line in f]
    return sources, targets


def load_counterfact_data(target_path, source_idx_path):
    with open(target_path) as f:
        targets = [line.strip() for line in f]
    with open(source_idx_path) as f:
        source_idx = [int(line.strip()) for line in f]
    return targets, source_idx


class MyDataset(Dataset):
    def __init__(self, tokenizer, data_prefix, max_input_length, max_target_length):
        sources, targets = load_data(data_prefix + '.source', data_prefix + '.target')
        tokenized_sources = tokenizer(sources, truncation=True, max_length=max_input_length)
        tokenized_targets = tokenizer(targets, truncation=True, max_length=max_target_length)
        self.input_ids = tokenized_sources['input_ids']
        self.attention_mask = tokenized_sources['attention_mask']
        self.labels = tokenized_targets['input_ids']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {
            'input_ids': self.input_ids[item],
            'attention_mask': self.attention_mask[item],
            'labels': self.labels[item],
        }


class MyCounterfactDataset(Dataset):
    def __init__(self, tokenizer, data_prefix, counterfact_data_prefix, max_input_length, max_target_length):
        sources, raw_targets = load_data(data_prefix + '.source', data_prefix + '.target')
        counterfact_targets, counterfact_source_idx = load_counterfact_data(counterfact_data_prefix + '.raw_target', counterfact_data_prefix + '.other')
        tokenized_sources = tokenizer(sources, truncation=True, max_length=max_input_length)
        tokenized_counterfact_targets = tokenizer(counterfact_targets, truncation=True, max_length=max_target_length)
        consistency_labels = self.get_consistency_labels(raw_targets, counterfact_targets, counterfact_source_idx, tokenized_counterfact_targets)

        self.input_ids = np.array(tokenized_sources['input_ids'], dtype=object)[counterfact_source_idx].tolist()
        self.attention_mask = np.array(tokenized_sources['attention_mask'], dtype=object)[counterfact_source_idx].tolist()
        self.labels = tokenized_counterfact_targets['input_ids']
        self.consistency_labels = consistency_labels

    def get_consistency_labels(self, raw_targets, counterfact_targets, all_source_idx, tokenized_counterfact_targets):
        all_consistency_labels = []
        raw_targets = np.array(raw_targets)[all_source_idx]
        for i, (raw_seq, counterfact_seq) in tqdm(enumerate(zip(raw_targets, counterfact_targets)), total=len(raw_targets)):
            match_flg = np.zeros(len(counterfact_seq))
            seq_matcher = difflib.SequenceMatcher(None, a=raw_seq, b=counterfact_seq)
            match_blocks = seq_matcher.get_matching_blocks()
            for blk in match_blocks:
                match_flg[blk.b: blk.b + blk.size] = 1

            consistency_label = []
            offsets = tokenized_counterfact_targets.encodings[i].offsets
            for off in offsets:
                label = 1 if sum(match_flg[off[0]: off[1]]) == off[1] - off[0] else 0
                consistency_label.append(label)
            all_consistency_labels.append(consistency_label)

        return all_consistency_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {
            'input_ids': self.input_ids[item],
            'attention_mask': self.attention_mask[item],
            'labels': self.labels[item],
            'consistency_labels': self.consistency_labels[item],
        }


@dataclass
class MyCounterfactDataCollator:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def _pad(self, features, name):
        item = [feature[name] for feature in features] if name in features[0].keys() else None
        if item is not None:
            max_length = max(len(l) for l in item)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_length - len(feature[name]))
                feature[name] = (feature[name] + remainder if padding_side == "right" else remainder + feature[name])

    def __call__(self, features):
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        self._pad(features, "labels")
        self._pad(features, "consistency_labels")

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
