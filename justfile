# cpu, cu128 or cu130
torch := "cu130"

default:
    @just --list

# Install dependencies, build diffvg, check the setup
setup: sync diffvg
    uv run khattat doctor

# --inexact keeps diffvg, which is not in the lockfile
sync:
    uv sync --extra {{torch}} --inexact

# Build diffvg against the local CUDA toolkit
diffvg *args:
    bash scripts/build_diffvg.sh {{args}}

# Format and autofix
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Lint and type check
check: lint typecheck

lint:
    uv run ruff format --check .
    uv run ruff check .

typecheck:
    uv run ty check src

# Khattat vs Word-as-Image on a word list
eval words="assets/eval_words.tsv" out="runs/eval" *args="":
    uv run python scripts/paper_eval.py --words {{words}} --out {{out}} {{args}}
