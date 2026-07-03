from __future__ import annotations
from pathlib import Path
from typing import List
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
        is_utf8: bool = True
    ) -> ObsDataFrame | None:
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
    is_utf8 : bool, default: True
        If True, the file is assumed to be encoded in UTF-8; otherwise, it is assumed to be in Big5 encoding.
    
    Returns
    -------
    ObsDataFrame | None
        A pandas DataFrame containing the observational data, or None if the file is empty or an error occurs.
    """
    def parse_datetime(datetime_str: str) -> pd.Timestamp:
        if datetime_str[-2:] == "24":
            datetime_str = datetime_str[:-2] + "00"
            return pd.to_datetime(datetime_str, format="%Y%m%d%H") + pd.to_timedelta("1 days")
        return pd.to_datetime(datetime_str, format="%Y%m%d%H")
    
    def drop_nan_and_keep_sta_info(data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna(subset=[col for col in data.columns if col not in ["# stno", "yyyymmddhh"]], how="all")
    
    def count_widths() -> list[int] | None:
        def count_widths_in_line(line: str, following_width: int) -> list[int]:
            return [6, 12] + [following_width] * int((len(line) - 18) / following_width)
        with open(path, "r", encoding=None if is_utf8 else "cp950") as file:
            for line in file:
                if line.startswith("#"):
                    return count_widths_in_line(line, 9 if is_utf8 else 7)            
    
    def read_fwf() -> pd.DataFrame:
        def extract_nans(nans: NanMode | SpecialValue | list[SpecialValue]) -> list[str]:
            if isinstance(nans, list):
                result = []
                for value in nans:
                    result.extend(extract_nans(value))
                return result
            else:
                return nans.utf8 if is_utf8 else nans.big5
        if mode is None:
            na_values = extract_nans(NanMode.AllEmpty)
        else:
            na_values = extract_nans(mode)    
        params: dict = {"na_values": na_values}
        if not is_utf8:
            params["encoding"] = "big5"
        return pd.read_fwf(
            path,
            widths=count_widths(),
            comment="*",
            **params
        ).astype({"# stno": str})
    print(f" R| {path.name}")
    result = read_fwf()
    if result.empty:
        print(f" E| {path.name} | Empty")
        return None
    result = drop_nan_and_keep_sta_info(result) if drop_nan else result
    result.insert(1, "datetime", result["yyyymmddhh"].astype(str).apply(parse_datetime))
    result.drop("yyyymmddhh", axis=1, inplace=True)
    return ObsDataFrame(result)

def load_folder(
    path: Path,
    mode: NanMode = NanMode.AllEmpty,
    max_threads: int = 4,
    drop_nan: bool = False,
    selected_cols: list[str] | None = None,
    station_number: str | None = None,
) -> ObsDataFrame | None:
    """
    Load all observational data files in a folder into a combined pandas DataFrame.

    Parameters
    ----------
    path : Path
        The path to the folder containing observational data files.
    mode : NanMode, default: NanMode.AllEmpty
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
    if selected_cols is None:
        selected_cols = ["TX01", "PP01", "PS01", "RH01", "WD01", "WD02"]
    datasets: list[dict[str, object]] = []
    datasets_lock = threading.Lock()
    print("==  Read Folder")
    print(f"==|==Begein| Mode: {mode.name} | Threads: {max_threads}")
    print(f"===========| Path: {path}")
    def process_file(file_path: Path):
        display_name = file_path.name
        try:
            dataset = load_file(file_path, mode, drop_nan)
        except UnicodeDecodeError:
            dataset = load_file(file_path, mode, drop_nan, is_utf8 = False)
        except Exception as e:
            print(f" E|2-{display_name}: {e}")
            raise
        finally:
            print(f"     Loaded| {file_path}")
        try:
            if dataset is None:
                return
            data_obj = dataset
            if station_number is not None:
                data_obj = dataset.station(station_number)
            if selected_cols is not None:
                data_obj = data_obj.get_item_with_time(selected_cols)

            with datasets_lock:
                datasets.append({"file_name": display_name, "data_obj": data_obj})
        except Exception as e:
            print(f" E|3-{display_name}: {e}")
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
        return None

    print(datasets_df)
    print("====  Finish   ==================")
    return pd.concat(datasets_df["data_obj"].values).obs