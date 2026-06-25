# GCP scale-up plan (constructor_causal messy-world campaign)

When seed counts / world grids grow past what the local box handles comfortably, run on a GCP
GPU VM. The code is already device-agnostic (`device.py`: auto `cuda > mps > cpu`) and the runner
(`run_seeds.py`) is **parallel + resumable + multi-GPU**, so this is a packaging + launch story,
not a code change.

## When it's worth it
- These models are tiny full-batch MLPs: one training ≈ 2–4 min, GPU only ~1.5× over one CPU
  core. So a **single** GPU on a **single** seed is NOT the win.
- The win is **throughput**: N seeds × V variants × W worlds is embarrassingly parallel.
  `run_seeds.py` runs one-seed-per-worker across all GPUs. With G GPUs you get ≈ G× over a
  single-GPU loop, on top of resume so nothing is recomputed.
- The bigger future win (one GPU, all seeds at once via `torch.func.vmap` over stacked params)
  is noted under "Next" — not built yet because it's only worth it at very large grids.

## One-time: provision a GPU VM
```bash
gcloud compute instances create chemworld-gpu \
  --zone <ZONE> --machine-type a2-highgpu-1g \
  --image-family common-cu121 --image-project deeplearning-platform-release \
  --accelerator type=nvidia-tesla-a100,count=1 --maintenance-policy TERMINATE --boot-disk-size 100GB
# (use count=4 + a2-highgpu-4g for a 4-GPU box; pick the accelerator your quota allows)
```

## Each run: package → upload → setup → launch
```bash
# LOCAL: package the code (no venv/results/git)
bash gcp/pack.sh                                   # -> /tmp/chemworld.tar.gz
gcloud compute scp /tmp/chemworld.tar.gz chemworld-gpu:~ --zone <ZONE>

# ON THE VM:
tar xzf chemworld.tar.gz && bash gcp/setup.sh      # CUDA torch + deps (one time)
bash gcp/launch.sh headline 30 8                   # 30 seeds, 8 workers, auto GPU count
bash gcp/launch.sh ablation 20 8 4                 # 20 seeds across 4 GPUs
```

## Resume / fetch
```bash
# resume after a kill/scale: re-run the SAME command -- finished seeds are skipped
bash gcp/launch.sh headline 50 8                   # extends 30 -> 50, only runs 30..49

# fetch results back
gcloud compute scp --recurse chemworld-gpu:~/ChemicalWorld/results ./ --zone <ZONE>
```

## Cost discipline
- **Stop the VM when idle**: `gcloud compute instances stop chemworld-gpu --zone <ZONE>`
  (A100s bill by the second while RUNNING). Delete when done.
- Estimate before launching: ~`(seeds × per-seed-min) / (gpus × workers-per-gpu)`.

## What the runner guarantees
- `--run {headline,ablation,harder}` — the three campaign jobs.
- `--seeds N` runs seeds `0..N-1`; **resumable** via `results/<run>.jsonl` (skip finished).
- `--workers W` parallel processes; `--gpus G` round-robins `CUDA_VISIBLE_DEVICES`.
- per-seed JSON checkpoint (seed, repr scores, readout R², rollout err/slope, success, final m3,
  failure) flushed as each seed lands; summary printed at the end.

## Next (only if grids get large)
Vectorize training across seeds (`torch.func.stack_module_state` + `vmap`) so one GPU trains all
seeds in ~one training's time (≈10–30×). Bigger refactor; deferred until the parallel-process
path is the bottleneck.
