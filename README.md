# SAR-SSL results (branch `results`)

Written by `scripts/collect_result.py` on the TAU cluster, not by hand. Code lives on `main`.

    results/<task>/<time>__<job_id>/     task: TDOA, T60, DRR, ABS, C50 or pretrain; time: the run's MMDDHHMM stamp
      summary.json                       task, model (from description.txt), date, seed, lr, frozen, status, metrics, ckpt_dir
      downstream_test_res*.json          raw per-trial results
      description.txt, config.json, pretext_test_res.json   when they exist
    results/summary_tables/*.md          tables written by code/gen_summary_table.py

`metrics` has one entry per test mode, named after the result file: `finetune`, `lineareval`, `scratchLOW`, their
`_real` variants, and `default` for the older `downstream_test_res.json`. Each entry has `lowest`, `mean`, `seed_std`
(population std across trials, as in `gen_summary_table.py`) and the raw `trials`. `metric_test` is an error, so lower is better.

`job_id` `runai` marks runs from the old RunAI server, now kept in `/moshe-data`. They have no git commit.
`ckpt_dir` is the path on `/moshe-data`, which only the nodes with Slurm feature `ib_dj` mount.
Files over 1 MB, checkpoints and audio are never stored here.
