#!/bin/bash
# Fetch the external offline model-based baselines at the commits methods/clap
# and methods/tdmpc2 were written against (third_party/ is not tracked).
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party
fetch() {  # NAME URL COMMIT
  [ -d third_party/$1 ] || git clone -q $2 third_party/$1
  git -C third_party/$1 fetch -q origin && git -C third_party/$1 checkout -q $3
  echo "$1 $(git -C third_party/$1 rev-parse --short HEAD)"
}
fetch c-lap https://github.com/marvinalles/c-lap.git e1616be
fetch tdmpc2 https://github.com/nicklashansen/tdmpc2.git e9f5932
