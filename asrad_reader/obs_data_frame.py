from dataclasses import dataclass, field
from functools import cached_property
from typing import overload
import pandas as pd

@dataclass(frozen = True)
class ObservePattern:
    """
    Represents an observation pattern for a specific station number.

    Attributes
    ----------
    station_number : int
        The station number associated with the observation pattern.
    start_time : pd.Timestamp
        The starting time of the observation pattern.
    end_time : pd.Timestamp
        The ending time of the observation pattern.
    enabled_hours : set[int]
        A set of integers representing the hours during which observations were enabled.
    """
    station_number: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    enabled_hours: set[int]
    length: pd.Timedelta = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'length', self.end_time - self.start_time)

@dataclass(frozen = True)
class NanPeriod:
    """
    Represents a period of NaN values for a specific station number.
    
    Attributes
    ----------
    station_number : str
        The station number associated with the NaN period.
    sequence_number : int
        The sequence number for the NaN period.
    start_time : pd.Timestamp
        The starting time of the NaN period.
    end_time : pd.Timestamp
        The ending time of the NaN period.
    length : pd.Timedelta
        The duration of the NaN period.
    """
    station_number: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    length: pd.Timedelta = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'length', self.end_time - self.start_time)

@pd.api.extensions.register_dataframe_accessor("obs")
class ObsDataFrame:
    """
    A pandas DataFrame accessor for handling observational data related to station numbers and timestamps.

    Attributes
    ----------
    station_list : set[str]
        A set of unique station numbers present in the DataFrame.
    
    Methods
    -------
    find_observe_patterns(station_number: str | None = None) -> list[ObservePattern] | dict[str, list[ObservePattern]]
        Finds observation patterns for a specific station number or for all station numbers.
    find_nan_periods(station_number: str | None = None, specify_columns: list[str] | None = None) -> list[NanPeriod] | dict[str, list[NanPeriod]]
        Finds periods of NaN values for a specific station number or for all station numbers, optionally specifying columns to check for NaN values.
    
    See Also
    --------
    pandas.DataFrame : The underlying DataFrame that this accessor is attached to.
    """
    def __init__(self, df: pd.DataFrame):
        self._df = df
        if "# stno" not in self._df.columns:
            raise AttributeError("DataFrame must contain a '# stno' column.")
        if "datetime" not in self._df.columns and not isinstance(self._df.index, pd.DatetimeIndex):
            raise AttributeError("DataFrame must contain a 'datetime' column or a DatetimeIndex.")
    
    @cached_property
    def station_list(self) -> set[str]:
        """
        Returns a set of unique station numbers from the DataFrame.
        
        Returns
        -------
        set[str]
            A set containing unique station numbers.
        """
        return set(self._df["# stno"].unique())
    
    def station(self, station_number: str) -> "ObsDataFrame":
        """
        Returns an ObsDataFrame filtered by the specified station number.

        Parameters
        ----------
        station_number : str
            The station number to filter by.

        Returns
        -------
        ObsDataFrame
            An ObsDataFrame containing only rows with the specified station number.
        """
        return ObsDataFrame(self._df[self._df["# stno"] == station_number])

    def _with_datetime_column(self) -> pd.DataFrame:
        if "datetime" in self._df.columns:
            return self._df.copy()

        frame = self._df.copy()
        frame = frame.reset_index()
        if "index" in frame.columns and "datetime" not in frame.columns:
            frame.rename(columns={"index": "datetime"}, inplace=True)
        return frame

    def get_item_with_time(self, cols: list[str]) -> pd.DataFrame:
        frame = self._with_datetime_column()
        target_cols = ["# stno"]
        if "datetime" in frame.columns:
            target_cols.append("datetime")
        target_cols.extend([col for col in cols if col in frame.columns and col not in target_cols])
        return frame[target_cols].copy()

    def drop_nan(self) -> pd.DataFrame:
        return self._df.dropna(subset=[col for col in self._df.columns if col not in ["# stno", "datetime"]], how="all").copy()
    
    @overload
    def find_observe_patterns(self, station_number: str) -> list[ObservePattern]:
        """
        Find observation patterns for a specific station number.

        Parameters
        ----------
        station_number : str
            The station number for which to find observation patterns.
        
        Returns
        -------
        list[ObservePattern]
            A list of ObservePattern instances corresponding to the specified station number.
        """
        ...
    
    @overload
    def find_observe_patterns(self) -> dict[str, list[ObservePattern]]:
        """
        Find observation patterns for all station numbers.

        Returns
        -------
        dict[str, list[ObservePattern]]
            A dictionary mapping station numbers to their corresponding lists of ObservePattern instances.
        """
        ...
    
    def find_observe_patterns(self,
            station_number: str | None = None
        ) -> list[ObservePattern] | dict[str, list[ObservePattern]]:
        if station_number is None:
            return {stno: self.find_observe_patterns(stno) for stno in self.station_list}

        frame = self._with_datetime_column()
        station_data = frame.loc[frame["# stno"] == station_number]
        if station_data.empty:
            return []

        daily_hours = {
            pd.Timestamp(date): frozenset(group["datetime"].dt.hour)
            for date, group in station_data.groupby(station_data["datetime"].dt.date)
        }

        if not daily_hours:
            return []

        patterns = []
        dates = list(daily_hours.keys())
        seg_start = dates[0]
        cur_hours = daily_hours[seg_start]

        for date in dates[1:]:
            hours = daily_hours[date]
            if hours != cur_hours:
                patterns.append(ObservePattern(
                    station_number=station_number,
                    start_time=seg_start,
                    end_time=date - pd.Timedelta("1d"),
                    enabled_hours=set(cur_hours),
                ))
                seg_start = date
                cur_hours = hours
        
        patterns.append(ObservePattern(
            station_number=station_number,
            start_time=seg_start,
            end_time=dates[-1],
            enabled_hours=set(cur_hours),
        ))
        return patterns
    
    @overload
    def find_nan_periods(self, station_number: str) -> list[NanPeriod]:
        """
        Find NaN periods for a specific station number.

        Parameters
        ----------
        station_number : str
            The station number for which to find NaN periods.

        Returns
        -------
        list[NanPeriod]
            A list of NanPeriod instances corresponding to the specified station number.
        """
        ...
    
    @overload
    def find_nan_periods(self) -> dict[str, list[NanPeriod]]:
        """
        Find NaN periods for all station numbers.

        Returns
        -------
        dict[str, list[NanPeriod]]
            A dictionary mapping station numbers to their corresponding lists of NanPeriod instances.
        """
        ...
    
    def find_nan_periods(self,
            station_number: str | None = None,
            specify_columns: list[str] | None = None
        ) -> list[NanPeriod] | dict[str, list[NanPeriod]]:
        if station_number is None:
            # Return a dictionary mapping station numbers to their NaN periods
            nan_periods_dict = {}
            for stno in self.station_list:
                nan_periods_dict[stno] = self.find_nan_periods(stno)
            return nan_periods_dict
        else:
            # Return a list of NaN periods for the specified station number
            results = []
            cols = specify_columns if specify_columns is not None else self._df.columns
            station_data = self.station(station_number)
            for col in cols:
                try:
                    is_nan = station_data[col].isnull()
                    # Group by consecutive non-NaN values to find NaN intervals
                    nan_groups = is_nan.groupby((is_nan != is_nan.shift()).cumsum())
                    for _, group in nan_groups:
                        if all(group):  # Check if the entire group is NaN
                            start_time = group.index[0]
                            end_time = group.index[-1]
                            results.append(NanPeriod(station_number, start_time, end_time))
                except NameError as e:
                    print(e)
            return results
        
    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns the underlying pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The underlying DataFrame.
        """
        return self._df.copy()

    def __getitem__(self, key):
        return self._df[key]
