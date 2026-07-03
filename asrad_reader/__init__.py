"""
This dataset is design for process CWA longtime observe data from https://asrad.pccu.edu.tw.
---
Created by wyork507. (contact information: https://wyork507.site)
"""

# asrad-reader package initialization

__version__ = "0.4.5"

from .nan_mode import NanMode
from .obs_data_frame import NanPeriod, ObsDataFrame, ObservePattern
from .loader import load_file, load_folder

__all__ = [
	"NanMode",
	"ObsDataFrame",
	"ObservePattern",
	"NanPeriod",
	"load_file",
	"load_folder",
]
