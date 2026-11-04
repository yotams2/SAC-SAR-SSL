# SAC-SAR-SSL

Fork of **SAR-SSL** (Yang et al., TASLP 2024: self-supervised spatial acoustic representation learning by cross-channel
signal reconstruction with a multi-channel Conformer), extended with the **Soft Acoustic Contrastive (SAC) /
feature-similarity loss**: continuous Gaussian-kernel similarity weights over analytical spatial/acoustic features
(DOA, T60, DRR, C50, room geometry, ...) pull embeddings of physically similar clips together. This is the spatial
predecessor of the SAC idea that `../sac-ssamba` applies to SSAMBA.

- The upstream README.md is the original SAR-SSL one (datasets, data generation and pretrain/downstream commands).
- [sac_loss_hyperparameter_optimization_guide.md](sac_loss_hyperparameter_optimization_guide.md): SAC loss formulation
  and the label-free sigma/tau selection method (alignment, uniformity, effective rank, weight entropy).
- Development was mostly done on RunAI (Aug to Sep 2026 commits). This cluster setup has only just started (see below).

## History: two machines

The project was developed on an **external RunAI server** (layout `/storage/yotam/SAR-SSL/SAR-SSL/code`, data next to it).
It is being moved to the **TAU power cluster** (Slurm). As of 2026-11-02, the cluster setup is only partly done:
- `setup_venv.sh` (venv at `/scratch200/yotams2/venvs/sac-sar-ssl`, gpuRIR built) and `generate_50k_samples.sbatch` exist
  but are **not committed**.
- `/scratch200/yotams2/data/` is **empty**: no WSJ0 source speech or simulated MicSig here yet.
- `code/opt.py` derives every path from `work_dir = <code>/../..` and then appends `/SAR-SSL/code`, `/SAR-SSL/data`,
  `/SAR-SSL/<exp_dir>`. This assumes the RunAI layout `<work_dir>/SAR-SSL/` (the repo folder named `SAR-SSL`), and data
  under `<work_dir>/data`. Here the repo is `~/SAC-SAR-SSL` and the home quota can't hold data/exp, so pass
  `--work-dir` or add an env-var override pointing at `/scratch200` before running anything.
- `../sac-s3prl` has a `sarssl` upstream that imports this code from `$SAR_SSL_PATH` (default is the RunAI path).

## Storage on this cluster

| Location | What | Notes |
|---|---|---|
| `~/SAC-SAR-SSL` (home) | this git repo (code only) | home quota is ~2 GB: no data, exp, checkpoints or venvs here |
| `/scratch200/yotams2/` | working copy: `venvs/sac-sar-ssl`, `data/` (intended for `SrcSig/wsj0`, `MicSig/simu`), logs | reachable from all nodes, large, but **scratch: may be wiped, not long-term storage**. Slow: avoid recursive scans/`du` |
| `/moshe-data/blaufer-data/yotams2/` | long-term copy of what came over from RunAI, including `SAC-SAR-SSL/exp` | **mounted on few nodes** (not the Jupyter node): reach it with `--constraint=ib_dj --exclude=compute-0-351`. Don't point training jobs at it; treat it as the backup |

Anything important that exists only on `/scratch200` (checkpoints, results) should be copied to `/moshe-data`.
The `.gitignore` excludes `data/`, `exp/`, `*.pt`/`*.ckpt`, `wandb/`, `metrics/` and `analysis_results/`, so none of
those are in git.

## Environment

- Its own venv, separate from sac-ssamba's: `/scratch200/yotams2/venvs/sac-sar-ssl`, created by `./setup_venv.sh`
  (`--force` rebuilds). It uses `--system-site-packages` on top of the cluster module, so always load the modules first:
  ```bash
  module load python/python-3.11_torch-2.2.2 CUDA/CUDA-12.2
  source /scratch200/yotams2/venvs/sac-sar-ssl/bin/activate
  ```
- gpuRIR is compiled into this venv (needed for data generation).

## Running jobs

- The interactive (Jupyter) node has 1 GB RAM and no GPU: `import torch` gets OOM-killed. Locally only do static
  checks (`python -m py_compile`, `bash -n`). Everything else goes through `sbatch`.
- Slurm account `public-blaufer_v2` (`--qos=public`). GPU jobs: `--partition=gpu-general-pool`, plus
  `--exclude=compute-0-18` (Blackwell sm_120, torch 2.2.2 has no kernels). CPU jobs: `power-general-shared-pool`.
- `gpu-general-pool` is **preemptible** (requeue). Long runs must be resumable, and use `#SBATCH --open-mode=append` so the
  requeued job doesn't truncate its log.
- `generate_50k_samples.sbatch` writes its logs to `/scratch200/yotams2/logs/` (it creates the folder).

## Code map (`code/`)

- `opt.py`: all CLI options and the directory layout (`dir()`). Loss switches: `--feature-sim-loss` (SAC),
  `--factorized_sac`, `--spectral-feature-sim-loss`, hierarchical room/DOA variants, `--obt`/`--hsic`, `--use_pcgrad`,
  `--diff-feat-selection`, `--fixed-feature-weights-path`, `--oracle_supcon` (upper bound with simulator ground truth).
- `model.py`, `learner.py`: MC-Conformer model, spatial/spectral split encoders, loss implementations.
  `common/`: Conformer, CRNN, Transformer building blocks.
- `run_pretrain.py`, `run_downstream.py`: pretext and downstream (tasks TDOA, T60, DRR, C50, ABS) training and testing.
  Runs are identified by a `--time` stamp (`MMDDHHMM`) and saved under `<exp>/pretrain/<time>`.
- `run_entire_flow.py`: pretrain, visualize, then all downstream tasks for one timestamp. `run_real_data_flow.py`: same
  for the real-data setting.
- `data_generation/`: gpuRIR simulation (`gen_simu.py`, `gen_simu_certain_room.py`) and real RIR / LOCATA preparation.
- `tuning/`: label-free sigma/tau and feature selection (Optuna, entropy search, multi-seed stability).
- `room_classification/`: probes on the latent space (room classifier, volume regressor).
- Analysis/visualization scripts at the top level (`compare_embeddings.py`, `visualize_embeddings.py`,
  `gen_summary_table.py`, ...). `tmp.sh` is a personal loop script, not part of the pipeline.
- `../code_v1`: original SAR-SSL v1 code (matches the paper). Reference only.
- `../tests`: pytest tests for the feature-similarity loss and spatial features.

## Results sync and the cloud Claude

I also work with Claude in the claude.ai app ("cloud Claude"). It cannot reach this cluster. It reads this repo from
GitHub (code on `main`, results on the `results` branch) and the Notion thesis archive. Anything not pushed is
invisible to it. The full rules are in `../sac-ssamba/CLAUDE.md`, section of the same name; this repo differs as follows.

- **Results** live on the branch `results`, in the worktree `~/SAC-SAR-SSL-results`:
  `results/<task>/<time>__<job_id>/summary.json` plus the raw json files. Never commit results to `main`. Format:
  `README.md` on that branch. `metric_test` is an error, so lower is better; `seed_std` is computed as in `gen_summary_table.py`.
- **Past runs** are in `/moshe-data/blaufer-data/yotams2/SAC-SAR-SSL/exp`, which the Jupyter node can't see. To report new ones,
  submit `sbatch scripts/collect_moshe.sbatch` from the repo root: it runs on a node that mounts `/moshe-data`
  (`--constraint=ib_dj`, minus compute-0-351), and it is safe to rerun.
- **No launch script reports by itself yet**, because no run has been launched on this cluster: only data generation is set up.
  When you write the first one, end it with `collect_result.py --exp_root <exp> --task <task> --time <time>` and
  `scripts/sync_results.sh`, and add the hook and post-job closer from `../sac-ssamba/scripts/` (`results_hook.sh`,
  `collect_after.sbatch`) so that failed and preempted runs are recorded too.
- **RunAI**: runs made there are invisible until copied here and collected. To report them from RunAI directly, the machine
  needs this repo's `results` worktree and a GitHub SSH key.
- **Start of every session**: `scripts/sync_results.sh`, `git pull` on main, read the Notion page "Agent Handoff" (under SAC Loss).
  **End of every session**: `scripts/sync_results.sh`, commit, and **push `main`**.

## Git

Remote `git@github.com:yotams2/SAC-SAR-SSL.git`. Push over SSH (HTTPS fails through the VS Code credential helper).
