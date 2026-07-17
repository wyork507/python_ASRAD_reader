from __future__ import annotations
from pathlib import Path
from typing import List, Literal
import concurrent.futures
import threading
import pandas as pd

from .nan_mode import NanMode
from .special_value import SpecialValue
from .obs_data_frame import ObsDataFrame

def load_file(
        path: Path,
        mode: NanMode | List[SpecialValue] | None = None,
        drop_nan: bool = False,
        encoding: Literal['utf-8', 'big5', 'cp950'] = 'utf-8'
    ) -> pd.DataFrame:
    """
    Load a single observational data file into a pandas DataFrame.

    Parameters
    ----------
    path : Path
        The path to the observational data file.
    mode : NanMode | List[SpecialValue], default: NanMode.AllEmpty
        The mode for handling NaN values. 
    drop_nan : bool, default: False
        If True, rows with all NaN values (except for station number and datetime) will be dropped.
    encoding : Literal['utf-8', 'big5', 'cp950'], default: 'utf-8'
        The encoding of the file.
    on_error : Literal['error', 'warn', 'skip'], default: "error"
        Determines how to handle errors during file loading:
        - "error": Raise an exception and stop processing.
        - "warn": Print a warning message and continue processing.
        - "skip": Silently skip the file and continue processing.
    Returns
    -------
    ObsDataFrame
        A pandas DataFrame containing the observational data
    
    Raises
    ------
    ValueError
        If the file is empty or contains no valid data.
    """
    def read_fwf() -> pd.DataFrame:
        def count_widths() -> list[int]:
            def count_widths_in_line(line: str, following_width: int) -> list[int]:
                return [6, 12] + [following_width] * int((len(line) - 18) / following_width)
            with open(path, "r", encoding=encoding) as file:
                for line in file:
                    if line.startswith("#"):
                        return count_widths_in_line(line, 9 if encoding == "utf-8" else 7) 
            raise ValueError(f"No header line found in file {path}.")  
        def extract_nans(nans: NanMode | SpecialValue | list[SpecialValue]) -> list[str]:
            if isinstance(nans, list):
                result = []
                for value in nans:
                    result.extend(extract_nans(value))
                return result
            else:
                return nans.utf8 if encoding == "utf-8" else nans.big5
        def params() -> dict:
            params: dict = {}
            params["na_values"] = extract_nans(mode) if mode is not None else extract_nans(NanMode.AllEmpty) + ["None"]
            if encoding != "utf-8":
                params["encoding"] = encoding
            return params
        
        return pd.read_fwf(
            path,
            widths=count_widths(),
            comment="*",
            **params()
        ).astype({
            "# stno": str,
            "yyyymmddhh": str
        })
    
    def validate_not_empty(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            raise ValueError(f"File {path} is empty, please check the file content.")
        return data
    
    def insert_datetime_col(data: pd.DataFrame) -> pd.DataFrame:
        def parse_datetime(datetime_str: str) -> pd.Timestamp:
            if datetime_str[-2:] == "24":
                datetime_str = datetime_str[:-2] + "00"
                return pd.to_datetime(datetime_str, format="%Y%m%d%H") + pd.to_timedelta("1 days")
            return pd.to_datetime(datetime_str, format="%Y%m%d%H")
        return data.assign(
            datetime=data["yyyymmddhh"].apply(parse_datetime)
        ).drop(columns=["yyyymmddhh"])
    
    def drop_nan_rows(data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna(
            subset=[col for col in data.columns if col not in ["# stno", "datetime"]],
            how="all"
        )
    
    def keep_all_rows(data: pd.DataFrame) -> pd.DataFrame:
        return data
    
    def arange_columns(data: pd.DataFrame) -> pd.DataFrame:
        columns_order = ["datetime", "# stno"] + [col for col in data.columns if col not in ["# stno", "datetime"]]
        return data[columns_order]
    
    print(f" R| {path.name}")
    
    return (read_fwf()
        .pipe(validate_not_empty)
        .pipe(insert_datetime_col)
        .pipe(drop_nan_rows if drop_nan else keep_all_rows)
        .pipe(arange_columns)
    )

def load_folder(
    path: Path,
    mode: NanMode | List[SpecialValue] | None = None,
    max_threads: int = 4,
    drop_nan: bool = False,
    selected_cols: list[str] | None = None,
    station_number: str | None = None,
) -> pd.DataFrame:
    """
    Load all observational data files in a folder into a combined pandas DataFrame.

    Parameters
    ----------
    path : Path
        The path to the folder containing observational data files.
    mode : NanMode | List[SpecialValue], default: NanMode.AllEmpty
        The mode for handling NaN values.
    max_threads : int, default: 4
        The maximum number of threads to use for loading files concurrently.
    drop_nan : bool, default: False
        If True, rows with all NaN values (except for station number and datetime) will be dropped.
    selected_cols : list[str], default: ["TX01", "PP01", "PS01", "RH01", "WD01", "WD02"]
        The list of columns to include in the final DataFrame.
    station_number : str, default: None
        If specified, only data for the given station number will be included in the final DataFrame.
    """
    if mode is None:
        mode = NanMode.AllEmpty
    if selected_cols is None:
        selected_cols = ["TX01", "PP01", "PS01", "RH01", "WD01", "WD02"]
    datasets: list[dict[str, object]] = []
    datasets_lock = threading.Lock()
    print("==  Read Folder")
    print(f"==|==Begein| Mode: {mode} | Threads: {max_threads}")
    print(f"===========| Path: {path}")
    
    def process_file(path: Path):
        try:
            try:
                dataset = load_file(path, mode, drop_nan)
            except UnicodeDecodeError:
                dataset = load_file(path, mode, drop_nan, encoding="big5")
            except Exception:
                raise    
            print(f"     Loaded| {path.name}")
        except ValueError:
            print(f"    Unloaded| {path} (empty or invalid data)")
            dataset = None
        except Exception as e:
            print(f" E|2-{path.name}: {e}")
            raise
        try:
            if dataset is None:
                return
            data_obj = dataset.obs.station(station_number) if station_number else dataset
            data_obj = data_obj.obs.get_items(selected_cols) if selected_cols else data_obj
            with datasets_lock:
                datasets.append({"file_name": path.name, "data_obj": data_obj})
        except Exception as e:
            print(f" E|3-{path.name}: {e}")
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(process_file, file_path)
            for file_path in path.rglob("*.txt")
            if file_path.is_file()
        ]
        concurrent.futures.wait(futures)

    print(f"==| Resort | Total {len(datasets)} file(s)")
    datasets_df = pd.DataFrame(data=datasets, columns=["file_name", "data_obj"]).sort_values("file_name")

    if datasets_df.empty:
        print("====  Error : No valid data files found in folder")

    print(datasets_df)
    print("====  Finish   ==================")
    return pd.concat(datasets_df["data_obj"].values)