"""
Pre-process DYAMOND files and save Parquet files to TAR archive.

NOTE: requires `pyarrow` to save the GeoDataFrames
"""

from lib import BASE_DIR_OUT_PRE

BASE_DIR_OUT_PRE.mkdir(exist_ok=True)
