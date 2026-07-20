English | [繁體中文](readme_ZhTW.md)

# About This Library
The `asrad-reader` is a small Python package for reading CWA long-term observational data from ASRAD (Atmospheric Science Research and Appliction Databank ,[link](https://asrad.pccu.edu.tw/)) and turning it into a pandas-friendly dataset.

This project is built around two ideas:
* Quick loading of many observation files with multi-threading
* Convenient time-series processing through a pandas `DataFrame` extension

The author is [wyork507](https://wyork507.site).

### Current Structure
```text
asrad-reader/
   __init__.py
   loader.py
   nan_mode.py
   obs_data_frame.py
   special_value.py
   py.typed
readme.md
readme_ZhTW.md
read_me.html
setup.py
```

### Requirements
* Python 3.11.7 or higher
* NumPy 1.26.2 or higher
* pandas 2.1.4 or higher

# Getting Started

## 1. Installation
For published releases, users should install the package from PyPI:
```bash
pip install asrad-reader
```

## 2. Importation
Import the package:
```python
from pathlib import Path
from asrad_reader import load_file, load_folder
```
## 3. Read a File

### For a sigle file
```python
file_path = Path("datas/20029999_cwb_hr/20021099.cwb_hr.txt")
df = load_file(file_path)
```
Here, the `load_file` supports variety of parameters, please see source code or IDE hint for details.
### For multiple files in folders
```python
folder_path = Path("datas")
df = load_folder(folder_path)
```
The `load_folder` reads all `.txt` files under the folder with multi-threading, also supported parameters.

## 4. DataFrame Accessor
The returned object is an `DataFrame`, so you can also use the registered accessor:
```python
df.obs.station("467490")
df.obs.get_item_with_time(["TX01", "PP01", "RH01"])
df.obs.find_observe_patterns()
df.obs.find_nan_periods()
```
See the `ObsDataFrame` class to know more method and usage.

## Notes
* `ASRAD_reader.__init__` exports `NanMode`, `SpecialValue`, `ObsDataFrame`, `ObservePattern`, `NanPeriod`, `load_file`, and `load_folder`.
* `ASRAD_reader` is intended to be used as a library, not as a standalone script.
* For any detail information in parameters, check your IDE hint or source code.

Last update: 2026/07/21.