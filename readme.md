## About This Repository
During my summer internship, I tried to read the station observational data from the Atmospheric Science Research
and Application Databank(大氣科學研究與應用資料庫，或稱大氣水文資料庫). However, I found it is difficult to access, so I
fix some bug and published the code I used.

By this reader, you can:
* Quick load ASRAD data with mutil-threads
* Useful format for timeseries analyze

Since this class is inherited from `pandas.DataFrame`, you can use its functions.

### Develop Environment
* NumPy 1.26.2 or higher
* pandas 2.1.4 or higher
* python 3.12.8 or higher

## Getting Started
1. Download the ASRAD_reader.py, and put into same folder with main.py
2. Import this reader as a normal libraries, see below:
    ```python3
    from ASRAD_reader import NanMode
    import ASRAD_reader as Reader
    ```

## How to use?
### Read a file
Use example below to input the file.
```python3
file_path = "datas/20029999_cwb_hr/20021099.cwb_hr.txt"
df = Reader.DataSet.read_file(file_path)
```
The `mode` parameter, is a enumeration named `NanMode`, which defines what values should be treated as NaN (Not a Number).
There are 4 specific mode: `ObsEmpty`, `AllEmpty`, `NotInObs`, and `AllValue`. Default is `NotInObs`.
```python3
df = Reader.DataSet.read_file(file_path, mode = NanMode.AllValue)
```
The `drop_nan` parameter,
if set to `True`, rows containing NaN values will be removed from the DataFrame. Defaults to `False`.
```python3
df = Reader.DataSet.read_file(file_path, drop_nan = True)
```
The `is_utf8` parameter, specifies whether the file is encoded in UTF-8 (`True`) or Big5 (`False`).
```python3
df = Reader.DataSet.read_file(file_path, is_utf8 = True)
```
The mode and its special value:

|   Mode   | Description                       |
|:--------:|:----------------------------------|
|`ObsEmpty`| Any reason without observation.   |
|`AllEmpty`| All but without trace.            |
|`NotInObs`| No data because no observation.   |
|`AllValue`| All special value cases.          |

### Read a folder contain files
Example below:
```python3
folder_path = "datas/"
df = Reader.DataSet.read_folder(folder_path)
```
Other parameter of `read_folder`:
* `mode`, optional, default is `NanMode.AllEmpty`.
* `max_threads`, optional, default is 4.
* `drop_nan`, optional, default is `True`.

Or only read specific columns:
```python3
folder_path = "datas/"
df = Reader.DataSet.read_folder_selected(folder_path)
```
Other parameter of `read_folder_selected`:
* `mode`, optional, default is `NanMode.AllEmpty`.
* `max_threads`, optional, default is 4.
* `drop_nan`, optional, default is `True`.
* `selected_cols`, optional, default is `["TX01", "PP01", "PS01", "RH01", "WD01", "WD02"]`.
* `station_number`, optional, default is 467490.

## Storage as .csv file
1. To normal .csv via DataFrame function:
   ```python3
   df.to_csv("datas.csv")
   ```
   For detail, please refer to [pandas document](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html).
2. To special .csv, which could load by this reader
   ```python3
   df.to_datasets_csv("datas.csv")
   ```
   And use the command below to load back.
   ```python3
   df = Reader.DataSet.read_dataset_csv("datas.csv")
   ```

## 
