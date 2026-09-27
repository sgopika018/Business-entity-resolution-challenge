# Business Entity Resolution — Unstop/Amazon ML Challenge

This project contains the complete scalable pipeline and the supplied challenge dataset. The data has millions of rows, so the production script uses a disk-backed SQLite index and streams Source 1 instead of loading the full dataset into RAM.

## Dataset

The supplied data is already in:

```text
dataset/train/
dataset/test/
```

## VS Code terminal commands

### Windows PowerShell
```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python src/scalable_entity_resolution.py --data-dir . --mode predict --threshold 0.62
python src/official_validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/scalable_entity_resolution.py --data-dir . --mode predict --threshold 0.62
python src/official_validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

The first run builds `entity_index.sqlite` from the test Source 2 and Source 3 files. It then streams every test Source 1 record and writes:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

If the command is interrupted after the index is complete, rerun the same command; it reuses the index. The output files are overwritten safely.

## Pipeline method

1. Normalize names and addresses with Unicode-aware case folding, punctuation removal, legal suffix reduction, and common address abbreviations.
2. Build exact indexes for normalized name, normalized address, country-name prefix, and country-address suffix.
3. Generate the final candidate set as the union of those high-precision blocking keys, capped at 500 records per key to prevent pathological common-key explosions.
4. Score candidates with exact normalized-field agreement plus token and character similarity.
5. Use a conservative default threshold of `0.62`, appropriate for the precision-heavy macro F0.5 metric.
6. Emit every Source 1 entity exactly once, including empty matches for singletons.

No external business lookup, geocoding, API, or internet enrichment is used.
