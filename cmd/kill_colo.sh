#!/bin/bash

# target_string="$1"
target_string="run_clm_colo_llama.py"

if [ -z "$target_string" ]; then
  echo "f"
  exit 1
fi


pids=$(pgrep -f "$target_string")

if [ -z "$pids" ]; then
  echo 'could not find run_clm_colo_llama.py'
  exit
fi

for pid in $pids; do
  echo "killing $pid"
  kill -9 $pid
done

echo "kill \"$target_string\" "