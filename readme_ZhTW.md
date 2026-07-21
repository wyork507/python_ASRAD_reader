[English](readme.md) | 繁體中文

# 關於套件
本套件（`asrad-reader`, [link](https://github.com/wyork507/python_asrad_reader)）是一個小型的 Python 套件，用於讀取大氣科學研究與應用資料庫（ASRAD, [link](https://asrad.pccu.edu.tw/)）中來自氣象署的長期測站觀測紀錄，轉換成 `Pandas`友善的格式。

主要目的在於提供：
* 使用多核心，快速讀取多個檔案
* 方便的時序處理功能

作者是 [wyork507](https://wyork507.site)。

### 專案結構
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

### 依賴需求
* Python 3.11.7 或以上
* NumPy 1.26.2 或以上
* pandas 2.1.4 或以上

# 開始使用

## 1. 安裝
可以透過 PyPI 安裝公開發行版:
```bash
pip install asrad-reader
```

## 2. 導入
導入本套件
```python
from pathlib import Path
from asrad_reader import load_file, load_folder
```
## 3. 讀檔

### 單一檔案
```python
file_path = Path("datas/20029999_cwb_hr/20021099.cwb_hr.txt")
df = load_file(file_path)
```
上面使用的 `load_file` 支援多種參數；關於參數用法，請讀原始碼或參考 IDE 提示。
### 多個檔案
```python
folder_path = Path("datas")
df = load_folder(folder_path)
```
此處使用之 `load_folder` 可以讀取在該目錄之下的所有檔案 `.txt` 檔案，並使用多個核心讀取，也有參數可以使用。

## 4. 從 DataFrame 存取
讀檔後，會回傳 `DataFrame` 格式，可以使用`.obs`將其轉換為`ObsDataFrame`。
```python
df.obs.station("467490")
df.obs.get_item_with_time(["TX01", "PP01", "RH01"])
df.obs.find_observe_patterns()
df.obs.find_nan_periods()
```
請參見 `ObsDataFrame` 類別以深入瞭解其包含之方法。

## 備註
* `ASRAD_reader.__init__`會導入`NanMode`、`SpecialValue`、`ObsDataFrame`、`ObservePattern`、`NanPeriod`、`load_file`和`load_folder`。
* `ASRAD_reader`是一種庫，而非獨立腳本。
* 對於各種參數的深入資訊，請參見IDE提示或者是原始碼。

上次更新：2026/07/21。