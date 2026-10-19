#!/bin/zsh
cd -- "${0:A:h}" || exit 1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gpilot-fork-mpl"
.venv/bin/python run_all.py
result=$?
if [[ "$result" == 2 ]]; then
  print '试算结束但待复核。请查看output/result_quality.csv。'
fi
read -r '?按回车关闭窗口。'
exit "$result"
