#!/usr/bin/env bash
#
# This will run our ablation experiment and generate the plot.

if (( $# < 1 )); then
  echo "usage: run-ablation.sh <cpu | gpu>"
  exit 1
fi

proj="$(cd "$(dirname "$0")"; cd ..; pwd)"
dev="$1"

n_runs=10
datasets=(
  jodie-lastfm
  # jodie-mooc
  # jodie-reddit
  # jodie-wiki
  # snap-email
  snap-msg
  # snap-reddit
)

numa_cmd=""
common_args="--model tgat --runs $n_runs"
if [[ "$dev" == "gpu" ]]; then
  common_args+=" --gpu 0"
fi

cd $proj
mkdir -p logs

echo
echo ">> running ablation: baseline"
echo "dataset,avg,std" > logs/ab-base.csv
for d in "${datasets[@]}"; do
  echo; $numa_cmd python inference.py $common_args -d $d --prefix ab-base --csv logs/ab-base.csv
done

echo
echo ">> running ab1: cache"
echo "dataset,avg,std" > logs/ab1.csv
for d in "${datasets[@]}"; do
  echo; $numa_cmd python inference.py $common_args -d $d --prefix ab1 --opt-cache --csv logs/ab1.csv
done

echo
echo ">> running ab2: cache+dedup"
echo "dataset,avg,std" > logs/ab2.csv
for d in "${datasets[@]}"; do
  echo; $numa_cmd python inference.py $common_args -d $d --prefix ab2 --opt-cache --opt-dedup --csv logs/ab2.csv
done

echo
echo ">> running ab3: cache+dedup+time"
echo "dataset,avg,std" > logs/ab3.csv
for d in "${datasets[@]}"; do
  echo; $numa_cmd python inference.py $common_args -d $d --prefix ab3 --opt-all --csv logs/ab3.csv
done

echo
echo ">> merging results"
python scripts/plot-ablation.py --merge $dev logs/ab-base.csv logs/ab1.csv logs/ab2.csv logs/ab3.csv

echo
echo ">> generating plot"
python scripts/plot-ablation.py $dev logs/ab-base.csv logs/ab1.csv logs/ab2.csv logs/ab3.csv

echo
echo ">> done"
