import os
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score
from models.data import MyCounterfactDataset, MyCounterfactDataCollator
from models.pegasus import MyPegasusForConditionalGeneration
from models.trainer import ConsistencyPrjTrainer
from models.utils import _is_json_serializable
from transformers import Seq2SeqTrainingArguments, HfArgumentParser, EarlyStoppingCallback
from transformers.models.pegasus import PegasusTokenizerFast


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    model_path: str = field(default='', metadata={"help": "Model path."})
    data_dir: str = field(default='', metadata={"help": "Data dir."})
    negative_data_dir: str = field(default='', metadata={"help": "Negative data dir."})
    max_input_length: int = field(default=1024, metadata={"help": "Max input length."})
    max_target_length: int = field(default=None, metadata={"help": "Max input length."})
    early_stopping_patience: int = field(default=None, metadata={"help": "Early stopping patience."})
    mask_ratio: float = field(default=None, metadata={"help": "Mask ratio."})
    mask_type: str = field(default=None, metadata={"help": "Mask type."})


def main():
    parser = HfArgumentParser(MyTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    args.label_names = ['labels', 'consistency_labels']

    args.output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump({k: v for k, v in args.__dict__.items() if _is_json_serializable(v)}, f, indent=2)
    if args.local_rank in [-1, 0]:
        print(args)

    tokenizer = PegasusTokenizerFast.from_pretrained(args.model_path)
    model = MyPegasusForConditionalGeneration.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.keys_to_ignore_at_inference = ['logits',
                                                'past_key_values',
                                                'decoder_hidden_states',
                                                'decoder_attentions',
                                                'cross_attentions',
                                                'encoder_last_hidden_state',
                                                'encoder_hidden_states',
                                                'encoder_attentions',
                                                'counterfact_past_key_values']

    if args.local_rank in [-1, 0]:
        print('Creating train dataset...')
    train_dataset = MyCounterfactDataset(
        tokenizer,
        os.path.join(args.data_dir, 'train'),
        os.path.join(args.negative_data_dir, args.mask_type, 'train'),
        args.max_input_length,
        args.max_target_length,
    )

    if args.local_rank in [-1, 0]:
        print('Creating valid dataset...')
    valid_dataset = MyCounterfactDataset(
        tokenizer,
        os.path.join(args.data_dir, 'valid'),
        os.path.join(args.negative_data_dir, args.mask_type, 'valid'),
        args.max_input_length,
        args.max_target_length,
    )

    data_collator = MyCounterfactDataCollator(tokenizer, model=model)
    trainer = ConsistencyPrjTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    trainer.train()
    trainer.state.save_to_json(os.path.join(args.output_dir, 'log.json'))


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=-1)
    pad_idx = (labels[1] == -100)
    accuracy = accuracy_score(y_true=labels[1][~pad_idx], y_pred=pred[~pad_idx])
    return {"acc": accuracy}


if __name__ == '__main__':
    main()
