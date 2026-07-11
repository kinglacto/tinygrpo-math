#!/usr/bin/env bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nodelist=node01
#SBATCH --job-name=tinygrpo-no-shaping
#SBATCH --output=/scratch/yashas.kotre/tinygrpo-ablations/logs/%x-%j.out
#SBATCH --error=/scratch/yashas.kotre/tinygrpo-ablations/logs/%x-%j.err

set -euo pipefail

cd "$HOME/RL"
source .venv-tinygrpo/bin/activate
run_root=/scratch/yashas.kotre/tinygrpo-ablations
mkdir -p "$run_root"/{hf,tmp,logs}
export HF_HOME="$run_root/hf"
export TMPDIR="$run_root/tmp"

common=(
  --model_name Qwen/Qwen2.5-1.5B-Instruct
  --eval_size 200
  --max_train_updates 500
  --group_size 4
  --train_batch_size 2
  --beta 0.05
  --format_reward 0
  --parseable_reward 0
  --temperature 0.7
  --top_p 0.95
  --checkpoint_every 500
  --save_model
)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/tinygrpo_math.py \
  --mode train \
  "${common[@]}" \
  --train_offset 0 \
  --train_size 500 \
  --learning_rate 5e-6 \
  --output_dir "$run_root/no_format_parseability_reward/stage1"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/tinygrpo_math.py \
  --mode train \
  "${common[@]}" \
  --init_model_path "$run_root/no_format_parseability_reward/stage1/final_model" \
  --train_offset 500 \
  --train_size 500 \
  --learning_rate 2e-6 \
  --output_dir "$run_root/no_format_parseability_reward/stage2"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/tinygrpo_math.py \
  --mode eval \
  --eval_model_path "$run_root/no_format_parseability_reward/stage2/final_model" \
  --eval_size 200 \
  --format_reward 0 \
  --parseable_reward 0 \
  --output_dir "$run_root/no_format_parseability_reward/eval200"
