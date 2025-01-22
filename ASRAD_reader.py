"""
This dataset is design for process CWA longtime observe data from https://asrad.pccu.edu.tw.
---
Created by wyork507. (contact information: https://wyork507.site)
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import concurrent.futures
import numpy as np
import pandas as pd
import threading


@dataclass(frozen = True)
class Encoding:
    Big5: [str]
    UTF8: [str]


class SpecialValue(Enum):
    # 儀器故障待修
    WaitFix = Encoding(["-9991"], ["-999.1"])
    # 資料累計於後
    InBelow = Encoding(["-9996"], ["-9.6", "-999.6"])
    # 因故障而無資料
    Trouble = Encoding(None, ["-9.5", "-99.5", "-999.5", "-9999.5"])
    # 因不明原因或故障而無資料(Big5) # 因不明原因而無資料(UTF8)
    Unknown = Encoding(["-9997"], ["-9.7", "-99.7", "-999.7", "-9999.7"])
    # 雨跡(Trace)
    OnTrace = Encoding(["-9998"], ["-9.8"])
    # 未觀測而無資料
    NoInObs = Encoding(["-9999"], ["None"])


class NanMode(Enum):
    """
        ObsEmpty : any reason without observation \n
        AllEmpty : all but without trace \n
        NotExist : no data because no observation
        AllValue : all special value \n
    """
    @staticmethod
    def _merge(*values: SpecialValue.value):
        big5 = []
        utf8 = []
        for val in values:
            big5 += val.Big5 or []
            utf8 += val.UTF8 or []
        return Encoding(big5, utf8)
    # Any reason without observation
    ObsEmpty = _merge(SpecialValue.NoInObs.value,
                      SpecialValue.WaitFix.value,
                      SpecialValue.Trouble.value,
                      SpecialValue.Unknown.value)
    # All but without trace
    AllEmpty = _merge(SpecialValue.NoInObs.value,
                      SpecialValue.WaitFix.value,
                      SpecialValue.Trouble.value,
                      SpecialValue.Unknown.value,
                      SpecialValue.InBelow.value)
    # No data because no observation
    NotInObs = _merge(SpecialValue.NoInObs.value)
    # All cases
    AllValue = _merge(SpecialValue.NoInObs.value,
                      SpecialValue.WaitFix.value,
                      SpecialValue.Trouble.value,
                      SpecialValue.Unknown.value,
                      SpecialValue.InBelow.value,
                      SpecialValue.OnTrace.value)


def find_all_folder_path(folder_path: str | Path) -> list[Path]:
    """
    Return a list of folder paths that contain files (excluding folders that only contain sub-folders).
    :param folder_path: The initial folder to search.
    :return: A list of folder paths containing files.
    """
    folder_path = Path(folder_path)
    path_list = []
    
    # Ensure the input path is a directory
    if not folder_path.is_dir():
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")
    
    for item in folder_path.iterdir():
        if item.is_file():
            # Skip system files like ".DS_Store"
            if item.suffix == ".DS_Store":
                continue
            # Add the folder to the list if a file is found
            path_list.append(folder_path)
            # Stop iterating over this folder if a file is found
        elif item.is_dir():
            # Recursively search subdirectories
            path_list.extend(find_all_folder_path(item))
    # Ensure no duplicate paths are returned
    return list(set(path_list))


class DataSet(pd.DataFrame):
    @staticmethod
    def _parse_datetime_with_24h(datetime_str):
        if datetime_str[-2:] == '24':
            datetime_str = datetime_str[:-2] + '00'  # replace '24' into '00'
            return pd.to_datetime(datetime_str, format = '%Y%m%d%H') + pd.to_timedelta('1 days')  # add a day
        else:
            return pd.to_datetime(datetime_str, format = '%Y%m%d%H')
    
    @staticmethod
    def _drop_nan(data):
        return data.dropna(subset = [col for col in data.columns if col not in ["# stno", "yyyymmddhh"]], how = "all")
    
    def __init__(self, data: pd.DataFrame, path = "N/A: Unknown") -> None:
        """
        :param data:
        :param path:
        """
        try:
            super().__init__(data)
        except Exception as e:
            print(e)
        self.__dict__["path"] = path
        try:
            self.__dict__["station_list"] = self["# stno"].unique()
        except Exception as e:
            print(e)
    
    @classmethod
    def transformation(cls, data: pd.DataFrame, path: str = "N/A: Transformed"):
        """
        Only for transform use. Danger.
        """
        return cls(data, path)
    
    @classmethod
    def read_file(cls, file_path: Path, mode: NanMode = NanMode.NotInObs, drop_nan: bool = False, is_utf8: bool = False):
        """
        To initialize from file's path.
        :param file_path:
        :param mode: ObsEmpty, AllEmpty, NotInObs, AllValue
        :param drop_nan:
        :param is_utf8:
        :return:
        """
        def _count_widths(path: Path):
            widths = None
            if is_utf8 is True:
                file = open(path, 'r')
            else:
                file = open(path, 'r', encoding = 'big5')
            for line in file:
                if line.startswith('#') and is_utf8:
                    widths = [6, 12] + [9] * int((len(line) - 18) / 9)
                    break
                elif line.startswith('#') and not is_utf8:
                    widths = [6, 12] + [7] * int((len(line) - 18) / 7)
                    break
            file.close()
            return widths
        
        print(f" R|{mode.name}| {file_path}")  # Debug statement
        try:
            if is_utf8:
                result = pd.read_fwf(file_path, widths = _count_widths(file_path),
                                     comment = '*', na_values = mode.value.UTF8)
            else:
                result = pd.read_fwf(file_path, widths = _count_widths(file_path), encoding = 'big5',
                                     comment = '*', na_values = mode.value.Big5)
            if result.empty:
                print(f" R| {str(file_path[22:28])} | Error: Empty")
                return None
            result = cls._drop_nan(result) if drop_nan else result
            # parse into datetime format
            result.insert(1, "datetime", result["yyyymmddhh"].astype(str).apply(cls._parse_datetime_with_24h))
            result.drop("yyyymmddhh", axis = 1, inplace = True)
            # then, set it as index
            result.set_index("datetime", inplace = True)
        except Exception as e:
            print(f" E|1-{str(file_path[22:28])}: {e}")  # Debug statement
            raise e
        else:
            return cls(result, file_path)
    
    @classmethod
    def read_folder(cls, folder_path: Path, mode = NanMode.AllEmpty, max_threads = 4, drop_nan: bool = True):
        """
        Reads multiple files from a folder into a DataSet, utilizing multithreading with direct addition.
        :param folder_path:
        :param mode:
        :param max_threads:
        :param drop_nan:
        :return:
        """
        datasets = []
        datasets_lock = threading.Lock()  # For synchronizing access to datasets
        print(f"==  Read Folder (no selected)")
        print(f"==|==Begein| Mode: {mode.name} | Threads: {max_threads}")
        print(f"===========| Path: {folder_path}")
        
        def process_file(_path):
            """
			Processes a single file and adds the resulting DataSet to the combined datasets.
			"""
            nonlocal datasets
            _disp_path = str(_path)[22:28]
            try:
                dataset = cls.read_file(_path, mode, drop_nan)
                print(f"     Loaded| {_path}")
                with datasets_lock:
                    datasets.append({
                        "file_name" : _disp_path,
                        "data_obj"  : dataset})
            except Exception as e:
                print(f" E|2-{_disp_path}: {e}")
                raise e
        
        with concurrent.futures.ThreadPoolExecutor(max_workers = max_threads) as executor:
            futures = [executor.submit(process_file, file_path)
                       for path in find_all_folder_path(folder_path)
                       for file_path in path.iterdir()
                       if file_path.is_file() and file_path.suffix == ".txt"]
            
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
            
        print(f"==| Resort | Total {len(datasets)} file(s)")
        datasets = pd.DataFrame(data = datasets, columns = ["file_name", "data_obj"]).sort_values("file_name")
        
        if datasets.empty is False:
            print(datasets)
            print(f"====  Finish   ==================")
            return cls(pd.concat(datasets["data_obj"].values), path = f"N/A: Combined, top path: {folder_path}")
        else:
            print(f"====  Error : No valid data files found in folder")
            return None
    
    @classmethod
    def read_folder_selected(cls, folder_path: Path, mode = NanMode.AllEmpty, max_threads = 4, drop_nan: bool = True,
                             selected_cols: list[str] = ["TX01", "PP01", "PS01", "RH01", "WD01", "WD02"],
                             station_number: int = 467490):
        """
        :param folder_path:
        :param mode:
        :param max_threads:
        :param drop_nan:
        :param selected_cols:
        :param station_number:
        :return:
        """
        datasets = []
        datasets_lock = threading.Lock()  # For synchronizing access to datasets
        print(f"==  Read Folder selected, station_number: {station_number})")
        print(f"==| Begein | Mode: {mode.name} | Threads: {max_threads}")
        print(f"           | Path: {folder_path}")
        print(f"===========| Rows: {selected_cols}")

        def process_file(_path):
            """
		    Processes a single file and adds the resulting DataSet to the combined datasets.
		    """
            nonlocal datasets
            _disp_path = str(_path)[22:28]
            try:
                dataset = cls.read_file(_path, mode, drop_nan)
                print(f"     Loaded| {_path}")
                with datasets_lock:
                    datasets.append({
                        "file_name": _disp_path,
                        "data_obj" : dataset.find_station(station_number).get_item_with_time(selected_cols)})
            except TypeError:
                dataset = cls.read_file(_path, mode, drop_nan, True)
                print(f"     Loaded| {_path}")
                with datasets_lock:
                    datasets.append({
                        "file_name": _disp_path,
                        "data_obj" : dataset.find_station(station_number).get_item_with_time(selected_cols)})
            except Exception as e:
                print(f" E|2-{_disp_path}: {e}")
                raise e
        
        with concurrent.futures.ThreadPoolExecutor(max_workers = max_threads) as executor:
            futures = [executor.submit(process_file, file_path)
                       for path in find_all_folder_path(folder_path)
                       for file_path in path.iterdir()
                       if file_path.is_file() and file_path.suffix == ".txt"]
            
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
        
        print(f"==| Resort | Total {len(datasets)} file(s)")
        datasets = pd.DataFrame(data = datasets, columns = ["file_name", "data_obj"]).sort_values("file_name")
        
        if datasets.empty is False:
            print(datasets)
            print(f"====  Finish   ==================")
            return cls(pd.concat(datasets["data_obj"].values), path = f"N/A: Combined, top path: {folder_path}")
        else:
            print(f"====  Error : No valid data files found in folder")
            return None
    
    @classmethod
    def read_dataset_csv(cls, file_path: Path):
        """
        
        :param file_path:
        :return:
        """
        if not file_path:
            raise ValueError("File name must be provided.")
        print(f"==  Read DataSet CSV")
        print(f"==| Begin | {file_path}")
        try:
            with open(file_path, "r", encoding = "utf-8") as f:
                path = f.readline().lstrip("#").rstrip("\n").split(" ")[1]
            
            # Read the rest of the file as data
            temper_data = pd.read_csv(file_path, skiprows = 2, na_values = "?")
            
            # Ensure datetime column is parsed correctly
            temper_data["datetime"] = pd.to_datetime(temper_data["datetime"])
            temper_data.set_index("datetime", inplace = True)
        except Exception as e:
            raise ValueError(f"Error reading dataset from CSV: {e}")
        else:
            print(f"====  Finish   ==================")
            return cls(temper_data, path)
    
    """
    Section:
        Station
    """
    class StationData(pd.DataFrame):
        """
        Represents data for a specific station, inheriting from pd.DataFrame.
        """
        def __init__(self, data: pd.DataFrame) -> None:
            """
            :param data:
            """
            try:
                super().__init__(data)
            except Exception as e:
                print(e)
        
        @classmethod
        def build(cls, data):
            """
            Creates a StationData object from a DataFrame.
            """
            return cls(data)
        
        @property
        def to_dataset(self):
            return DataSet.transformation(self.value())
        
        @property
        def drop_nan(self) -> DataSet.StationData:
            """
            Drops rows with all NaN values (except for station and datetime).
            """
            data = self.copy()  # Create a copy to avoid modifying the original
            data.dropna(
                subset = [col for col in data.columns if col not in ["# stno"]],
                how = "all",
                inplace = True,
            )
            return self.build(data)  # Return a new StationData object
        
        def get_item_with_time(self, cols: list[str]) -> DataSet.StationData | None:
            """
            :param cols:
            :return: data with time and station information
            """
            try:
                return self.build(self[["# stno"] + cols])
            except Exception as e:
                print(e)
                return None
        
        def __getitem__(self, item):
            """
            Allows accessing data using the [] operator.
            """
            return super().__getitem__(item)
        
        def __repr__(self):
            return super().__repr__()
        
        def __str__(self):
            return "StationData: \n" + super().__str__()
    
    def find_station(self, number: int) -> StationData | None:
        """
        Filters data by station number
        :param number: which station number you want to get
        """
        return self.StationData(self.loc[self["# stno"] == number].copy())
    
    @property
    def taichung_station(self) -> StationData | None:
        return self.find_station(467490)
    
    @property
    def taipei_station(self) -> StationData | None:
        return self.find_station(466920)
    
    @property
    def station_list(self) -> np.ndarray:
        """
        :return: number of all stations in this data
        """
        return self.__dict__["station_list"]
    
    @property
    def daily_average(self) -> pd.DataFrame:
        """
        1.create a new dataframe all the time of this dataset, delta t is day
        2.then, avrage it value and fill in
        :return:
        """
        daily_resample = self.copy()
        daily_resample = (daily_resample
                          .groupby("# stno")
                          .resample("D")
                          .mean()
                          .reset_index("datetime"))
        daily_resample.set_index("datetime", inplace = True)
        daily_resample.index = daily_resample.index.to_period("D")
        return daily_resample
    
    @property
    def mouth_average(self) -> pd.DataFrame:
        """
        1.create a new dataframe all the time of this dataset, delta t is mouth
        2.call day_average() to retrive the average of days
        3.then, avrage it value and fill in
        :return:
        """
        mouth_resample = self.copy()
        mouth_resample = (mouth_resample
                          .groupby("# stno")
                          .resample("M")
                          .mean()
                          .reset_index("datetime"))
        mouth_resample.set_index("datetime", inplace = True)
        mouth_resample.index = mouth_resample.index.to_period("M")
        return mouth_resample
    
    @property
    def find_observe_period(self) -> pd.Series:
        """
        1.to count the number of its non-NA rows in same hour per day
        2.then, return it statics result
        :return:
        """
        hourly_counts = self.groupby(self.index.hour).size()
        return hourly_counts
    
    def check_nan(self, cols: list[str]) -> pd.DataFrame:
        """
        Checks for NaN values in specified columns and returns a DataFrame with NaN intervals.
        
        -- Return DataFrame formation:
            - "Column": The name of the column with NaNs.
            - "Start Time": The start time of the NaN interval.
            - "End Time": The end time of the NaN interval.
            - "Duration": The duration of the NaN interval.
        :param cols:
        :return: A DataFrame with columns include: "Column", "Start Time", "End Time", "Duration"
        
        """
        results = []
        for col in cols:
            try:
                is_nan = self[col].isnull()
                # Group by consecutive non-NaN values to find NaN intervals
                nan_groups = is_nan.groupby((is_nan != is_nan.shift()).cumsum())
                for _, group in nan_groups:
                    if all(group):  # Check if the entire group is NaN
                        start_time = group.index[0]
                        end_time = group.index[-1]
                        duration = end_time - start_time
                        results.append({
                            "Column"    : col,
                            "Start Time": start_time,
                            "End Time"  : end_time,
                            "Duration"  : duration
                        })
            except NameError as e:
                print(e)
        return pd.DataFrame(results)
    
    def get_item_with_time(self, cols: list[str]) -> DataSet.StationData | None:
        """
        :param cols:
        :return: data with time and station information
        """
        try:
            return self.build(self[["# stno", "datetime"] + cols])
        except Exception as e:
            print(e)
            return None
    
    @property
    def path(self) -> str:
        """
        :return: the path of this parse file
        """
        return self.__dict__["path"]
    
    def to_datasets_csv(self, storage_name: str, *args, **kwargs):
        """
        Saves the DataSet to a CSV file, including metadata about the path and station list.

        :param storage_name: The name of the file to save the data to.
        :param args: Additional arguments for the pandas to_csv method.
        :param kwargs: Additional keyword arguments for the pandas to_csv method.
        """
        if not storage_name:
            raise ValueError("Storage name must be provided.")
        
        metadata = [f"#file_path {self.path}\n",
                    f"#station_list {",".join(map(str, self.station_list))}"]

        try:
            # Write metadata and data to the CSV file
            with open(storage_name, "w", encoding="utf-8") as f:
                f.writelines(metadata)
                f.write("\n")  # Separate metadata from data
                self.to_csv(f, na_rep="?", *args, **kwargs)
            print(f"==| SaveFile | {storage_name}")
        except Exception as e:
            raise IOError(f"Error saving dataset to CSV: {e}")
    
    def __add__(self, other):
        if isinstance(other, DataSet):
            try:
                return DataSet(pd.concat([self, other], ignore_index = True), path = "N/A: Added")
            except Exception as e:
                print(e)
        else:
            raise TypeError("The other isn't DataSet type.")
    
    def __getitem__(self, item) -> pd.DataFrame:
        return super().__getitem__(item)
    
    def __repr__(self):
        return super().__repr__()
    
    def __str__(self):
        return "DataSet from " + str(self.__dict__["path"]) + ": \n" + super().__str__()
