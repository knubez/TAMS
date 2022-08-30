"""
Try the MOSA runner for preprocessed files
"""
from pathlib import Path

from tams.mosa import run_wrf_preproced

files = sorted(Path("~/Downloads/mosa-pre-sample").expanduser().glob("tb_rainrate*.parquet"))

df = run_wrf_preproced(files)
