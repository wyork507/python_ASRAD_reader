"""
This dataset is design for process CWA longtime observe data from https://asrad.pccu.edu.tw.
---
Created by wyork507. (contact information: https://wyork507.site)
"""

# ASRAD_reader package initialization

__version__ = "0.4.1"

from ASRAD_reader.NanMode import NanMode
from ASRAD_reader.Dataset import DataSet

__all__ = ['NanMode', 'DataSet']
