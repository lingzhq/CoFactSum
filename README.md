# CoFactSum

This repository contains the official implementation for the COLING'2025 paper "Enhancing Factual Consistency in Text Summarization via Counterfactual Debiasing". The paper's link and citation information will be updated soon.

> Despite significant progress in abstractive text summarization aimed at generating fluent and informative outputs, how to ensure the factual consistency of generated summaries remains a crucial and challenging issue. In this study, drawing inspiration from advancements in causal inference, we construct causal graphs to analyze the process of abstractive text summarization methods and identify intrinsic causes of factual inconsistency, specifically language bias and irrelevancy bias, and we propose **CoFactSum**, a novel framework that mitigates the causal effects of these biases through counterfactual estimation for enhancing the factual consistency of the generated content. **CoFactSum** provides two counterfactual estimation strategies, including Explicit Counterfactual Masking, which employs a dynamic masking approach, and Implicit Counterfactual Training, which utilizes a discriminative cross-attention mechanism. Besides, we propose a Debiasing Degree Adjustment mechanism to dynamically calibrate the level of debiasing at each decoding step. Extensive experiments conducted on two widely used summarization datasets demonstrate the effectiveness and advantages of the proposed **CoFactSum** in enhancing the factual consistency of generated summaries, outperforming several baseline methods.

## Installation

### Environment Setup

It is recommended to use a `Python 3.9` environment and install the following important version dependencies:

- `torch=1.7.1`
- `transformers=4.8.1`

After creating and activating your virtual environment, install the other dependencies:

```sh
pip install -r requirements.txt
```

### Data Preparation

- **Base Model**: Use the google/pegasus-xsum [model](https://huggingface.co/google/pegasus-xsum) and google/pegasus-cnn_dailymail [model](https://huggingface.co/google/pegasus-cnn_dailymail) from Hugging Face.
- **Datasets**: [XSum Dataset](https://github.com/EdinburghNLP/XSum) and [CNN DailyMail Dataset](https://github.com/abisee/cnn-dailymail)
- **Usage**: Place the datasets in the `./dataset` folder.

## Implementation

Here are the specific usage steps, note that please replace $DEVICE_ID with the actual GPU device ID you are using.
1. Generate Summaries from PEGASUS

```sh
bash ./scripts/gen_pegasus.sh $DEVICE_ID
```

2. Conduct Training in the Implicit Counterfactual Training (ICT) Module

```sh
bash ./scripts/train_pegasus_topk_attn.sh $DEVICE_ID
```

3. Conduct Training in the Debiasing Degree Adjustment (DDA) Module

```sh
bash ./scripts/train_pegasus_consistency_prj.sh $DEVICE_ID
```

4. Conduct Counterfactual Debiasing

```sh
bash ./scripts/gen_pegasus_counterfact.sh $DEVICE_ID
```

5. Evaluate Various Baselines

```sh
bash ./scripts/eval_[ccgs, cliff, cofactsum, corr, pegasus, spancopy, unl].sh $DEVICE_ID
```


