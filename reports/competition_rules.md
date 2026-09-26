# Competition Rules — Amazon ML Challenge 2026 (Business Entity Resolution)

Extracted during READ-ONLY RECONNAISSANCE (STAGE 0) on 2026-09-25.
Every item below is traced to a local official source file. Anything not present in
the local files is marked **UNKNOWN — MUST VERIFY** (never inferred).

## Sources inspected

| # | File | What it contains |
|---|------|------------------|
| S1 | `student_resource/README.md` | Official problem statement, file formats, output format, evaluation metric, constraints, academic-integrity rules, tips |
| S2 | `student_resource/Documentation_template.md` | Required methodology write-up structure |
| S3 | `student_resource/utils/validate_submission.py` | Official submission validator (authoritative for formatting rules) |
| S4 | `day 1.pdf` / `day 1.odt` | Internal baseline-strategy notes (NOT official rules; treated as guidance only) |
| S5 | `promts.odt` | Internal prompt playbook (NOT official rules) |

No other rule document (no separate `RULES.md`, no PDF rulebook) exists in the project directory.

## 1. Problem definition (S1)

- Source 1 is the deduplicated reference source. For **every** S1 entity, find **all**
  matching records in Source 2 and Source 3.
- An S1 entity may match **zero, one, or many** records.
- Records share no common identifier; only `business_name`, `business_address`, `country`
  are available.

## 2. Input files (S1)

All tab-separated (`.tsv`) with header `entity_id  business_name  business_address  country`.

| File | Rows (incl. header) | Data rows |
|------|--------------------:|----------:|
| `dataset/train/train_source1.tsv` | 2,206,822 | 2,206,821 |
| `dataset/train/train_source2.tsv` | 5,034,617 | 5,034,616 |
| `dataset/train/train_source3.tsv` | 5,285,604 | 5,285,603 |
| `dataset/train/train_ground_truth.tsv` | 2,206,822 | 2,206,821 |
| `dataset/test/test_source1.tsv` | 1,732,545 | 1,732,544 |
| `dataset/test/test_source2.tsv` | 4,887,274 | 4,887,273 |
| `dataset/test/test_source3.tsv` | 5,082,317 | 5,082,316 |

- Ground truth columns: `source1_entity_id`, `matched_entity_ids` (comma-separated,
  **empty when no match**).
- **Train countries: US (1,323,633), India (883,188). No France.**
- **Test countries: US (663,106), India (809,986), France (259,452).**
  France is an unseen country → pipeline must treat `country` as an open string label.

## 3. Output format (S1, enforced by S3)

Two TSV files in `output/`:

1. `matching_results.tsv` — **the only leaderboard-scored file**
   - Columns exactly: `source1_entity_id`, `matched_entity_ids`
   - Exactly one row per test S1 entity (1,732,544 rows + header)
   - `matched_entity_ids` = comma-separated, no duplicates, only `S2-`/`S3-` IDs that
     exist in the test set; **empty string** for singletons
   - No duplicate `source1_entity_id` rows
2. `candidate_pairs.tsv` — **not scored**, but required in the final zip
   - Columns exactly: `source1_entity_id`, `candidate_entity_ids`
   - Must be the **last** candidate set fed to the model (the exact inference input);
     final matches should be a subset of it
   - Same one-row-per-S1 / no-duplicate rules

Final package: `<team_name>_submission.zip` containing `output/`,
`code/business_entity_resolution/{src/,README.md,requirements.txt}` and the filled
`Documentation_template.md` (S1).

Validator command (S1/S3):
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## 4. Evaluation metric (S1)

```
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
```
- Computed **per Source 1 entity**, then **macro-averaged over ALL S1 entities**.
- Singletons count: correct empty prediction = 1.0; any prediction on a true singleton = 0.0.
- Precision is weighted 2× over recall → false merges are penalised heavily.
- Worked example in S1: predicted `[S2-47,S2-193,S3-812]` vs truth `[S2-47,S3-812]` → F0.5 = 0.714.

## 5. Constraints and prohibitions (S1)

| Rule | Source |
|------|--------|
| Output must pass validation or it is not evaluated; `SCORED` status shows the F0.5 | S1 §Constraints |
| `matched_entity_ids` may only reference test-set S2/S3 IDs; no S1 self-matches | S1 §Constraints |
| Every test S1 entity must appear; missing entities → rejection | S1 §Constraints |
| Duplicate ID-list entries or duplicate `source1_entity_id` rows → rejection | S1 §Constraints |
| Final model must be **MIT / Apache-2.0 licensed** and **≤ 8 billion parameters** | S1 §Constraints #5 |
| **STRICTLY PROHIBITED: external data lookup** — no commercial ER APIs, no government registries, no geocoding APIs, no internet data augmentation. Violation = immediate disqualification; code is reviewed. | S1 §Academic Integrity |
| Fair play: only the provided training/test data | S1 §Academic Integrity |
| Public LB = subset of test; private LB = remainder; final ranking uses **private** LB. Do not train on LB feedback. | S1 §Leaderboard Information |

## 6. Items NOT found locally — UNKNOWN, MUST VERIFY on the Portal

- **Submission deadline / challenge end date** — UNKNOWN — MUST VERIFY
- **Number of leaderboard submissions allowed per day/total** — UNKNOWN — MUST VERIFY
- **Team size / registration rules** — UNKNOWN — MUST VERIFY
- **Whether LLM/AI-assisted code authoring is explicitly permitted** — the local README
  does not address it; only external *data lookup* is prohibited. UNKNOWN — MUST VERIFY
  (assumption for now: AI-assisted *code* is acceptable, external *data* is not)
- **Prize / credit details, scoring schedule** — UNKNOWN — MUST VERIFY

## 7. Conflict resolution

Where anything in the agent prompt conflicts with S1/S3, **S1/S3 win**. Notable
differences already observed:

- The prompt's sample submission filename `submissions/baseline_v1.tsv` is **not** the
  official name → official names are `output/matching_results.tsv` and
  `output/candidate_pairs.tsv`.
- The prompt does not mention `candidate_pairs.tsv`; the official rules **require** it
  in the final zip and define it as the final pre-model candidate set.
- The prompt assumes a ~10 GB dataset; measured size is **2.4 GB**.
