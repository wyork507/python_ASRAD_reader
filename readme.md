## About This Repository
`ASRAD_reader` is a small Python package for reading CWA long-term observational data from ASRAD and turning it into a pandas-friendly dataset.

This project is built around two ideas:
* quick loading of many observation files with multi-threading
* convenient time-series processing through a pandas `DataFrame` extension

The main public entry points are:
* `ASRAD_reader.load_file`
* `ASRAD_reader.load_folder`
* `ASRAD_reader.NanMode`
* the `obs` DataFrame accessor on returned datasets

## Current Structure
```text
ASRAD_reader/
   __init__.py
   loader.py
   nan_mode.py
   obs_data_frame.py
   special_value.py
main.py
readme.md
read_me.html
setup.py
test_read/
   _run_notebook_check.py
   read_folder_test.ipynb
```

## Development Environment
* Python 3.11.7 or higher
* NumPy 1.26.2 or higher
* pandas 2.1.4 or higher

## Installation
For published releases, users should install the package from PyPI:
```python3
pip install ASRAD_reader
```

For local development in this repository, install in editable mode from the repository root:
```python3
pip install -e .
```

## Getting Started
Import the package and the NaN mode enum:
```python3
from pathlib import Path

from ASRAD_reader import NanMode, load_file, load_folder
```

## Read a File
```python3
file_path = Path("datas/20029999_cwb_hr/20021099.cwb_hr.txt")
df = load_file(file_path)
```

`load_file` supports these parameters:
* `mode`: controls which special values are treated as NaN. Default is `NanMode.AllEmpty`.
* `drop_nan`: removes rows whose data columns are all missing. Default is `False`.
* `is_utf8`: `True` for UTF-8, `False` for Big5. Default is `True`.

Available `NanMode` values:

| Mode | Description |
|:--|:--|
| `ObsEmpty` | Missing because of observation-related reasons |
| `AllEmpty` | All special values |
| `NotInObs` | Missing because the station was not observed |

## Read a Folder
```python3
folder_path = Path("datas")
df = load_folder(folder_path)
```

`load_folder` reads all `.txt` files under the folder with multi-threading.

Supported parameters:
* `mode`: default `NanMode.AllEmpty`
* `max_threads`: default `4`
* `drop_nan`: default `False`
* `selected_cols`: columns to keep, default `None` means keep the default observation columns
* `station_number`: filter a specific station, default `None`

## DataFrame Accessor
The returned object is an `ObsDataFrame`, so you can also use the registered accessor:
```python3
df.obs.station("467490")
df.obs.get_item_with_time(["TX01", "PP01", "RH01"])
df.obs.find_observe_patterns()
df.obs.find_nan_periods()
```

The accessor expects a `# stno` column and either a `datetime` column or a `DatetimeIndex`.

## Notes
* `ASRAD_reader.__init__` exports `NanMode`, `ObsDataFrame`, `ObservePattern`, `NanPeriod`, `load_file`, and `load_folder`.
* `ASRAD_reader` is intended to be used as a library, not as a standalone script.
