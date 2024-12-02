import torch
import argparse
import os
import json
from datetime import datetime
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from rouge_score.scoring import BootstrapAggregator
from rouge_score.io import _write_aggregates_to_csv
from tqdm import tqdm
from transformers.models.pegasus import PegasusTokenizerFast, PegasusForConditionalGeneration
from factsumm import FactSumm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--num_beams', default=None, type=int)
    parser.add_argument('--max_input_length', default=1024, type=int)
    parser.add_argument('--max_length', default=None, type=int)
    parser.add_argument('--min_length', default=None, type=int)
    parser.add_argument('--length_penalty', default=None, type=float)
    parser.add_argument('--no_repeat_ngram_size', default=None, type=int)
    parser.add_argument('--split_sent', action='store_true')
    parser.add_argument('--num_test_samples', default=None, type=int)
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    device = 'cuda'
    args.gpu_ids = os.environ['CUDA_VISIBLE_DEVICES']
    print(args)

    with open(os.path.join(args.data_dir, 'test.source')) as f:
        sources = [line.strip() for line in f]
    with open(os.path.join(args.data_dir, 'test.target')) as f:
        targets = [line.strip() for line in f]

    tokenizer = PegasusTokenizerFast.from_pretrained(args.model_path)
    model = PegasusForConditionalGeneration.from_pretrained(args.model_path)

    if device == 'cuda': model.cuda()
    model.eval()
    print('Model loaded')

    # Generate
    all_outputs = []
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(sources), args.batch_size)):
            batch_sources = sources[batch_start: batch_start + args.batch_size]
            inputs = tokenizer(batch_sources, return_tensors="pt", truncation=True, max_length=args.max_input_length, padding=True).to(device)
            outputs = model.generate(**inputs,
                                     num_beams=args.num_beams,
                                     max_length=args.max_length,
                                     min_length=args.min_length,
                                     length_penalty=args.length_penalty,
                                     no_repeat_ngram_size=args.no_repeat_ngram_size)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            all_outputs.extend(decoded)

    with open(os.path.join(args.output_dir, 'formatted-test.txt'), 'w') as f:
        for output in all_outputs:
            f.write(output + '\n')

    # Evaluate
    with open(os.path.join(args.output_dir, 'formatted-test.txt')) as f:
        all_outputs = [line.strip().replace('<n>', ' ') for line in f]

    # -> ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    aggregator = BootstrapAggregator()
    for s, t in zip(all_outputs, targets):
        if args.split_sent:
            s = '\n'.join(sent_tokenize(s))
            t = '\n'.join(sent_tokenize(t))
        aggregator.add_scores(scorer.score(s, t))
    results = aggregator.aggregate()
    _write_aggregates_to_csv(os.path.join(args.output_dir, 'rouge_scores.csv'), results)

    # -> FactSumm
    factsumm_scorer = FactSumm()
    results = factsumm_scorer(sources[:args.num_test_samples], all_outputs[:args.num_test_samples], device=device)
    print('FactSumm results: {}'.format(results))
    num_samples = str(args.num_test_samples) if args.num_test_samples is not None else 'f'
    with open(os.path.join(args.output_dir, 'factsumm_scores_{}.json'.format(num_samples)), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
