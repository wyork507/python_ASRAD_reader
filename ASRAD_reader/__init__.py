"""
This dataset is design for process CWA longtime observe data from https://asrad.pccu.edu.tw.
---
Created by wyork507. (contact information: https://wyork507.site)
"""

# ASRAD_reader package initialization

__version__ = "0.4.1"

from ASRAD_reader.nan_mode import NanMode
from ASRAD_reader.obs_data_frame import NanPeriod, ObsDataFrame, ObservePattern
from ASRAD_reader.loader import load_file, load_folder

__all__ = [
	"NanMode",
	"ObsDataFrame",
	"ObservePattern",
	"NanPeriod",
	"load_file",
	"load_folder",
]
