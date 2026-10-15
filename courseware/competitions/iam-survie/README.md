# iam-survie

The three "Survie à 2 mois" challenges of the course *IA pour la médecine*
(DFGSM2), one per session, on prod ML-Arena. Each is a copy of challenge 172
("2-Month Survival Prediction": file_v1, SUPPORT2 critically-ill patients,
`outcome` alive/dead at 2 months, accuracy, `submission.csv` =
`patient_id,outcome`), so each session gets its own leaderboard. The ids are in
`state.json` (`s1`, `s2`, `s3`).

| File | Role |
| --- | --- |
| `create.py` | Idempotent create / verify / benchmark / start / publish script, one table entry per session |
| `state.json` | The created challenge ids and benchmark scores (a re-run never creates a second copy) |
| `source_snapshot.json` | Challenge 172 as read at creation; `y_test.csv`, the benchmark file and the agent template stored as sha256 only (the test labels are private) |
| `overview.md` | French overview shared by the three challenges (`__SESSION_TITLE__`, `__SESSION_BLOCK__` filled by `create.py`) |
| `session_s1.md` … `session_s3.md` | The session-specific paragraph of each overview |

What a copy of 172 does and does not carry over (checked by `create.py`):

- carried: configuration, evaluation, environment (`env.py`, `y_test.csv`),
  machine, config values, tags, submission layout, cover image;
- not carried: the overview (replaced by the French one), the datasets (the copy
  has none: `create.py` recreates 172's dataset and uploads its `X_train.csv`,
  `X_test.csv`, `y_train.csv`, then checks them byte for byte), and the
  benchmark `submission.csv` (the copy has a header-only stub: `create.py`
  uploads 172's always-`alive` file and checks the run scores exactly the share
  of `alive` in `y_test`, 0.594419).

Before anything is started, step 7 checks that the data the overview links (the
course repo `racousin/ai_for_medicine`, `ia-medecine/donnees/support2/`) matches
the scorer: hosted `X_test.csv` ids = `y_test.csv` ids, hosted
`y_train_survie.csv` = the served `y_train.csv`. It fails otherwise, leaving the
challenges hidden and not started.

Run with the released SDK 3.0.0 (prod does not run the 4.x API):

```bash
PYTHONPATH=<path to mlarena-sdk 3.0.0> python create.py          # all sessions
PYTHONPATH=<path to mlarena-sdk 3.0.0> python create.py s2       # one session
```

Keys come from `courseware/.env`: `MLARENA_CREATOR_API_KEY` (creator routes) and
`MLARENA_STUDENT_API_KEY` (read-only visibility check). Attaching the challenges
to the course is done separately with the teacher key.
