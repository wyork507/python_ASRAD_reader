# Copilot AI Coding Agent Instructions for ASRAD_reader

## Project Overview
- This project provides a toolkit for reading and processing CWA (Central Weather Administration) long-term observation data, especially from the ASRAD database.
- Core logic is in the `ASRAD_reader` package, which is designed to be used as a library and is built on top of `pandas.DataFrame` for seamless timeseries analysis.
- The main entry points are `ASRAD_reader.Dataset.py` (class `DataSet`) and `ASRAD_reader.NanMode.py` (enum `NanMode`).

## Key Components
- `ASRAD_reader/Dataset.py`: Implements the `DataSet` class, which extends `pandas.DataFrame` and provides methods for reading, transforming, and analyzing ASRAD data. Includes multi-threaded folder reading and station-specific data extraction.
- `ASRAD_reader/NanMode.py`: Defines the `NanMode` enum for handling special missing value encodings, using logic from `ASRAD_reader/SpecialValue.py` and `special_value.json`.
- `ASRAD_reader/SpecialValue.py` and `special_value.json`: Centralize the mapping of special values (e.g., -9999, -999.1) for different encodings (Big5, UTF8).
- `setup.py`: Standard Python packaging, with dependencies on numpy and pandas. Project metadata is maintained here.

## Usage Patterns
- Always use `DataSet.read_file()` or `DataSet.read_folder()` to load data. Do not instantiate `DataSet` directly with raw data unless transforming.
- Use `NanMode` to control which special values are treated as NaN. Example: `DataSet.read_file(path, mode=NanMode.AllValue)`.
- For station-specific data, use `DataSet.find_station(station_number)` or the provided properties (`taichung_station`, `taipei_station`).
- All data operations are designed to be compatible with pandas workflows.

## Developer Workflows
- Install dependencies with `pip install -e .` in the project root.
- No custom build steps; standard setuptools applies.
- No test suite is present by default—add tests in a `tests/` directory if needed.
- For new special value encodings, update both `SpecialValue.py` and `special_value.json`.

## Project Conventions
- All file reading expects CWA/ASRAD text file formats, with encoding handled via the `mode` and `is_utf8` parameters.
- Multi-threaded reading is implemented using `concurrent.futures.ThreadPoolExecutor`.
- All public API methods are documented with docstrings and print debug information for traceability.
- The package is intended for use as a library, not as a standalone script.

## Example Usage
```python
from ASRAD_reader import DataSet, NanMode
file_path = "datas/20029999_cwb_hr/20021099.cwb_hr.txt"
df = DataSet.read_file(file_path, mode=NanMode.AllValue, drop_nan=True)
```

## Key Files
- `ASRAD_reader/Dataset.py`, `ASRAD_reader/NanMode.py`, `ASRAD_reader/SpecialValue.py`, `ASRAD_reader/special_value.json`
- `setup.py`, `readme.md`

---
If any conventions or workflows are unclear, please ask for clarification or check the latest `README.md`.
