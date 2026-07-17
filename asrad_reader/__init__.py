"""
This dataset is design for process CWA long-term observational data from https://asrad.pccu.edu.tw.
---
Created by wyork507. (contact information: https://wyork507.site)
"""

# asrad-reader package initialization

__version__ = "0.4.6"

from .nan_mode import NanMode
from .special_value import SpecialValue
from .obs_data_frame import NanPeriod, ObsDataFrame, ObservePattern
from .loader import load_file, load_folder

__all__ = [
	"NanMode",
	"SpecialValue",
	"ObsDataFrame",
	"ObservePattern",
	"NanPeriod",
	"load_file",
	"load_folder",
]
