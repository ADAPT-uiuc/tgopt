#!/usr/bin/env bash
#
# This will run our main experiment and generate the plot.

if (( $# < 1 )); then
  echo "usage: run-exp.sh <cpu | gpu>"
  exit 1
fi

proj="$(cd "$(dirname "$0")"; cd ..; pwd)"
dev="$1"

n_runs=10
datasets=(
  jodie-lastfm
  jodie-mooc
  jodie-reddit
  jodie-wiki
  snap-email
  snap-msg
  snap-reddit
)

numa_cmd=""
common_args="--model tgat --runs $n_runs"
if [[ "$dev" == "gpu" ]]; then
  common_args+=" --gpu 0"
fi

cd $proj
mkdir -p logs

echo
echo ">> running baseline"
echo "dataset,avg,std" > logs/exp-base.csv
for d in "${datasets[@]}"; do
  echo; $numa_cmd python inference.py $common_args -d $d --prefix exp-base --csv logs/exp-base.csv
done

echo
echo ">> running tgopt"
echo "dataset,avg,std" > logs/exp-opt.csv
for d in "${datasets[@]}"; do
  echo; $numa_cmd python inference.py $common_args -d $d --prefix exp-opt --opt-all --csv logs/exp-opt.csv
done

echo
echo ">> generating plot"
python scripts/plot-exp.py $dev logs/exp-base.csv logs/exp-opt.csv

echo
echo ">> done"
