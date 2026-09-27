# Business Entity Resolution Methodology

## Methodology used

The solution is a supervised, precision-oriented entity-resolution pipeline. Training labels are converted into positive and negative record pairs. A class-balanced logistic-regression classifier scores candidate pairs using interpretable similarity features. The decision threshold is selected using a held-out-style validation run over the labelled training set.

All processing uses only the supplied challenge files. No external databases, APIs, geocoding, web search, or identity lookup is used.

## Data preparation and normalization

Business names and addresses are lowercased, punctuation-normalized, tokenized, and transformed using common legal and address abbreviations. Examples include `corporation` to `corp`, `private` to `pvt`, `limited` to `ltd`, `street` to `st`, and `road` to `rd`. Both token-preserving and compact alphanumeric forms are retained. Country is treated as an open string field; no country whitelist is used.

## Candidate generation / blocking

For every Source 1 record, candidates are generated from the union of:

1. Exact normalized business-name key.
2. Exact normalized address key.
3. Country plus a name prefix key.
4. Country plus an address suffix key.
5. Character TF-IDF nearest neighbors over the combined normalized name and address, using character n-grams from 2 through 5.

The last candidate set is the set written to `output/candidate_pairs.tsv` and is exactly the set scored by the matching stage. This gives a high recall ceiling while avoiding an all-pairs comparison.

## Model architecture and feature engineering

Each candidate pair has the following features:

- Token Jaccard, containment, and Dice similarity for the business name.
- Character n-gram Jaccard for the compact business name.
- Token Jaccard, containment, and Dice similarity for the address.
- Character n-gram Jaccard for the compact address.
- Country equality.
- Exact compact-name equality.
- Exact compact-address equality.

A class-balanced logistic regression is trained on candidate pairs derived from the labelled training records. The model probability is used as a match confidence.

## Threshold and singleton handling

Because F0.5 weights precision more heavily than recall, the default inference threshold is conservative (`0.62`). The training evaluation command prints F0.5 across several thresholds so the operator can select the strongest threshold for the supplied data. Source 1 records with no candidate above threshold are emitted with an empty match list, preserving singleton predictions.

## Reproducibility

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/entity_resolution.py --data-dir . --mode train-evaluate
python src/entity_resolution.py --data-dir . --mode predict --threshold 0.62
python src/validate_submission.py dataset/test output
```

The pipeline is deterministic: normalization is deterministic, TF-IDF and nearest-neighbor generation have no random training step, and negative sampling uses a fixed seed.

## Limitations and audit notes

The model is intentionally self-contained and challenge-compliant. It does not use external information. A stronger final threshold should be selected from the validation printout after the real Unstop files are installed. Any changes to blocking or threshold should be documented and the validator should be run before submission.
