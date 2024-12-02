import torch
import argparse
import os
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer
from rouge_score.scoring import BootstrapAggregator
from rouge_score.io import _write_aggregates_to_csv
from qafacteval import QAFactEval
from factsumm import FactSumm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--pred_dir', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--max_length', default=None, type=int)
    parser.add_argument('--split_sent', action='store_true')
    parser.add_argument('--num_test_samples', default=None, type=int)
    parser.add_argument('--spancopy', action='store_true')
    args = parser.parse_args()

    if args.spancopy:
        data = torch.load(os.path.join(args.data_dir, 'test.pt'))
        sources = [x['document'] for x in data]
        targets = [x['summary'] for x in data]
    else:
        with open(os.path.join(args.data_dir, 'test.source')) as f:
            sources = [line.strip() for line in f]
        with open(os.path.join(args.data_dir, 'test.target')) as f:
            targets = [line.strip() for line in f]

    # Evaluate
    if args.spancopy:
        all_outputs = []
        out_files = sorted(os.listdir(args.pred_dir), key=lambda x: int(x.split('.')[0]))
        for out in out_files:
            with open(os.path.abspath(os.path.join(args.pred_dir, out))) as f:
                cur = f.readline()
            cur = ' '.join(word_tokenize(cur.strip())[:args.max_length])
            all_outputs.append(cur)
    else:
        with open(os.path.join(args.output_dir, 'formatted-test.txt')) as f:
            all_outputs = [line.strip().replace('<n>', ' ') for line in f]
            all_outputs = [' '.join(word_tokenize(cur)[:args.max_length])
                           for cur in all_outputs]

    # -> ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    aggregator = BootstrapAggregator()
    for s, t in zip(all_outputs, targets):
        if args.split_sent:
            s = '\n'.join(sent_tokenize(s))
            t = '\n'.join(sent_tokenize(t))
        aggregator.add_scores(scorer.score(s, t))
    results = aggregator.aggregate()
    _write_aggregates_to_csv(os.path.join(args.output_dir, 'rouge_scores_cc.csv'), results)

    # -> QAFactEval
    kwargs = {"cuda_device": 0, "use_lerc_quip": True,
              "verbose": True, "generation_batch_size": 1,
              "answering_batch_size": 1, "lerc_batch_size": 1}
    model_folder = "../QAFactEval/models/"
    metric = QAFactEval(
        lerc_quip_path=f"{model_folder}/quip-512-mocha",
        generation_model_path=f"{model_folder}/generation/model.tar.gz",
        answering_model_dir=f"{model_folder}/answering",
        lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
        lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
        **kwargs,
    )
    results = metric.score_batch_qafacteval(sources[:args.num_test_samples],
                                            [[x] for x in all_outputs[
                                             :args.num_test_samples]],
                                            return_qa_pairs=True)
    full_scores = [min(res[0]['qa-eval']['lerc_quip'], 5) / 5 for res in
                   results]
    score = sum(full_scores) / len(full_scores)
    save_res = {'score': score,
                'full_scores': full_scores}
    num_samples = str(args.num_test_samples) if args.num_test_samples is \
                                                not None else 'f'
    with open(os.path.join(args.output_dir, 'qafe_scores_{}.json'.format(
            num_samples)), 'w') as f:
        json.dump(save_res, f, indent=2)

    # -> FactSumm
    factsumm_scorer = FactSumm()
    results = factsumm_scorer(sources[:args.num_test_samples], all_outputs[:args.num_test_samples], device=device)
    print('FactSumm results: {}'.format(results))
    num_samples = str(args.num_test_samples) if args.num_test_samples is not None else 'f'
    with open(os.path.join(args.output_dir, 'factsumm_scores_{}.json'.format(num_samples)), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
