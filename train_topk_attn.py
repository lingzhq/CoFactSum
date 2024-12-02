import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from models.data import MyDataset
from models.pegasus_topk_attn import MyPegasusTopkAttn
from models.trainer import TopkAttnTrainer
from models.utils import _is_json_serializable
from transformers import Seq2SeqTrainingArguments, HfArgumentParser
from transformers.models.pegasus import PegasusTokenizerFast
from transformers.data.data_collator import DataCollatorForSeq2Seq


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    model_path: str = field(default='', metadata={"help": "Model path."})
    data_dir: str = field(default='', metadata={"help": "Data dir."})
    max_input_length: int = field(default=1024, metadata={"help": "Max input length."})
    max_target_length: int = field(default=None, metadata={"help": "Max input length."})
    mask_ratio: float = field(default=None, metadata={"help": "Mask ratio for positive attn positions."})


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
    model = MyPegasusTopkAttn.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))

    if args.local_rank in [-1, 0]:
        print('Creating train dataset...')
    train_dataset = MyDataset(
        tokenizer,
        os.path.join(args.data_dir, 'train'),
        args.max_input_length,
        args.max_target_length,
    )

    if args.local_rank in [-1, 0]:
        print('Creating valid dataset...')
    valid_dataset = MyDataset(
        tokenizer,
        os.path.join(args.data_dir, 'valid'),
        args.max_input_length,
        args.max_target_length,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = TopkAttnTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.state.save_to_json(os.path.join(args.output_dir, 'log.json'))


if __name__ == '__main__':
    main()
