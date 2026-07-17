from dataclasses import dataclass, field
from typing import Iterator, Literal, overload
from datetime import date, timedelta
import pandas as pd

@dataclass(frozen = True, kw_only=True)
class ObservePattern:
    """
    Represents an observation pattern for a specific station number.

    Attributes
    ----------
    station_number : int
        The station number associated with the observation pattern.
    observation_count : int
        The number of observation hours in the pattern.
    duration : int
        The length of the observation pattern in days, calculated as end_date - start_date + 1.
    start_date : date
        The starting date of the observation pattern.
    end_date : date
        The ending date of the observation pattern.
    observed_hours : tuple[int, ...]
        A tuple of integers representing the hours during which observations were enabled.
    """
    station_number: str
    observation_count: int = field(init=False)
    duration: int = field(init=False)
    start_date: date
    end_date: date
    observed_hours: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, 'observation_count', len(self.observed_hours))
        object.__setattr__(self, 'duration', (self.end_date - self.start_date + timedelta(days=1)).days)

@dataclass(frozen = True)
class NanPeriod:
    """
    Represents a period of NaN values for a specific station number and columns.
    
    Attributes
    ----------
    station_number : str
        The station number associated with the NaN period.
    column: str
        The column that is considered for identifying NaN values.
    start_time : pd.Timestamp
        The starting time of the NaN period.
    end_time : pd.Timestamp
        The ending time of the NaN period.
    missing_count : int
        The number of missing data points in the NaN period.
    duration : pd.Timedelta
        The duration of the NaN period.
    """
    station_number: str
    column: str
    missing_count: int
    duration: pd.Timedelta = field(init=False)
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    def __post_init__(self):
        object.__setattr__(self, 'duration', self.end_time - self.start_time)

@pd.api.extensions.register_dataframe_accessor("obs")
class ObsDataFrame:
    """
    A pandas DataFrame accessor for handling observational data related to station numbers and timestamps.
    """
    def __init__(self, df: pd.DataFrame):
        if "datetime" not in df.columns:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise AttributeError(
                    "DataFrame must contain a 'datetime' column "
                    "or use a DatetimeIndex."
                )

            df = df.reset_index(
                names="datetime"
            )
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            raise TypeError(
                "'datetime' must have a datetime64 dtype."
            )
        self._df = df
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns the underlying pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The underlying DataFrame.
        """
        return self._df.copy()

    @property
    def stations(self) -> set[str]:
        """
        Returns a set of unique station numbers from the DataFrame.
        
        Returns
        -------
        set[str]
            A set containing unique station numbers.
        """
        return set(self._df["# stno"].unique())
    
    def station(self,
            station_number: str,
            *,
            on_error: Literal["raise", "ignore"] = "raise"
        ) -> "ObsDataFrame":
        """
        Returns an ObsDataFrame filtered by the specified station number.

        Parameters
        ----------
        station_number : str
            The station number to filter by.
        on_error : Literal["raise", "ignore"] = "raise", optional
            How to handle the case where a station number is not found.
        
        Returns
        -------
        ObsDataFrame
            An ObsDataFrame containing only rows with the specified station number.
        
        Raises
        ------
        ValueError
            If the specified station number is not found in the DataFrame.
        ValueError
            If the on_error parameter is not one of the allowed values ("raise" or "ignore").
        """
        if on_error not in ["raise", "ignore"]:
            raise ValueError(f"Invalid value for on_error: {on_error}. Must be 'raise' or 'ignore'.")
        if station_number not in self.stations and on_error == "raise":
            raise ValueError(f"Station number '{station_number}' not found in DataFrame.")
        else:
            return ObsDataFrame(self._df[self._df["# stno"] == station_number])

    def get_items(self, cols: list[str]) -> pd.DataFrame:
        """
        Returns a DataFrame containing only the specified columns, along with the 'datetime' and '# stno' columns.

        Parameters
        ----------
        cols : list[str]
            A list of column names to include in the returned DataFrame.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified columns, along with 'datetime' and '# stno'.
        """
        target_cols = ["datetime", "# stno"]
        target_cols.extend(
            col for col in cols
            if col in self._value_columns
        )
        return self._df[target_cols].copy()
    
    @property
    def drop_nan(self) -> pd.DataFrame:
        """
        Returns a DataFrame with rows dropped where all columns (except for 'datetime' and '# stno') are NaN.
        """
        return self._df.dropna(subset=self._value_columns, how="all").copy()
    
    def __get_value_columns(self, df: pd.DataFrame) -> list[str]:
        return [col for col in df.columns if col not in ["# stno", "datetime"]]
    
    @property
    def _value_columns(self) -> list[str]:
        return self.__get_value_columns(self._df)

    def _processed_data(self,
            station_number: str,
            *,
            columns: list[str] | None = None,
            drop: bool = True
        ) -> pd.DataFrame:
        def identify_nan(df: pd.DataFrame) -> pd.DataFrame:
            from .special_value import SpecialValue as sv
            replacement = {key: pd.NA
                for case in (sv.all_cases() - {sv.OnTrace})
                for key in (case.big5 + case.utf8 + case.numerical)
            }
            df[self._value_columns] = df[self._value_columns].replace(replacement)
            return df
        def drop_nan(df: pd.DataFrame) -> pd.DataFrame:
            return df.dropna(subset=self._value_columns, how="all")
        # Filter the DataFrame for the specified station number and columns, then identify and drop NaN values
        df = self.station(station_number)
        df = df.get_items(columns) if columns is not None else df.to_dataframe()
        return df.pipe(identify_nan).pipe(drop_nan) if drop else df.pipe(identify_nan)
        
    @overload
    def find_observe_patterns(self,
            station_number: str,
            *,
            columns: list[str] | None = None,
            on_error: Literal["raise", "ignore"] = "raise"
        ) -> list[ObservePattern]:
        ...
    
    @overload
    def find_observe_patterns(self,
            station_number: list[str] | None = None,
            *,
            columns: list[str] | None = None,
            on_error: Literal["raise", "ignore"] = "raise"
        ) -> dict[str, list[ObservePattern]]:
        ...

    def find_observe_patterns(self,
            station_number: list[str] | str | None = None,
            *,
            columns: list[str] | None = None,
            on_error: Literal["raise", "ignore"] = "raise"
        ) -> list[ObservePattern] | dict[str, list[ObservePattern]]:
        """
        Returns a list of observation patterns for the specified station number(s).

        Parameters
        ----------
        station_number : str | list[str] | None
            The station number(s) to filter by. If None, patterns for all stations will be returned.
        columns : list[str] | None, optional
            A list of column names to include in the analysis. If None, all columns will be used.
        on_error : Literal["raise", "ignore"] = "raise", optional
            How to handle the case where a station number is not found. If "raise", a ValueError will be raised. If "ignore", an empty list will be returned for that station.
        
        Returns
        -------
        list[ObservePattern] | dict[str, list[ObservePattern]]
            A list of ObservePattern objects for the specified station number(s), or a dictionary mapping station numbers to lists of ObservePattern objects.
        
        Raises
        ------
        ValueError
            If the specified station number is not found in the DataFrame and on_error is set to "raise".
        ValueError
            If the on_error parameter is not one of the allowed values ("raise" or "ignore").
        """
        def extract_datetime(df: pd.DataFrame) -> pd.DataFrame:
            df["datetime"] -= pd.Timedelta(hours=1)
            df["date"] = df["datetime"].dt.date
            df["hour"] = df["datetime"].dt.hour + 1
            return df.copy()
        def pattern_by_date(df: pd.DataFrame) -> pd.DataFrame:
            def hours_pattern(hours: pd.Series) -> tuple[int, ...]:
                return tuple(sorted(hours.unique()))
            return df.groupby("date").agg(
                hours=("hour", hours_pattern),
            ).reset_index()
        def identify_segments(df: pd.DataFrame) -> pd.DataFrame:
            same_segment = (
                df["hours"].eq(df["hours"].shift()) # check if the hours are the same as the previous row
                & df["date"].diff().eq(pd.Timedelta(days=1)) # check if the date is consecutive to the previous row
            )
            df["segment_id"] = (~same_segment).cumsum() # create a new segment id for each group of consecutive dates with the same hours
            return df
        # Main logic
        if on_error not in ["raise", "ignore"]: # validate on_error parameter
            raise ValueError(f"Invalid value for on_error: {on_error}. Must be 'raise' or 'ignore'.")
        if station_number is None:
            return {stno: self.find_observe_patterns(stno, columns=columns, on_error=on_error)
                    for stno in self.stations}
        elif isinstance(station_number, list):
            return {stno: self.find_observe_patterns(stno, columns=columns, on_error=on_error)
                    for stno in station_number}
        try:
            station_data = self._processed_data(station_number, columns=columns, drop=True)
        except ValueError as e:
            if on_error == "raise":
                raise e
            else:
                return []
        daily_hours = (station_data
            .pipe(extract_datetime)
            .pipe(pattern_by_date)
            .pipe(identify_segments)
        )
        return [
            ObservePattern(
                station_number  = station_number,
                start_date      = group["date"].iloc[0],
                end_date        = group["date"].iloc[-1],
                observed_hours   = group["hours"].iloc[0]
            ) for _, group in daily_hours.groupby("segment_id")
        ]
    
    @overload
    def find_nan_periods(self,
            station_number: str,
            *,
            columns: list[str] | None = None,
            on_error: Literal["raise", "ignore"] = "raise",
            dropna: bool = True
        ) -> list[NanPeriod]:
        ...
    
    @overload
    def find_nan_periods(self,
            station_number: list[str] | None = None,
            *,
            columns: list[str] | None = None,
            on_error: Literal["raise", "ignore"] = "raise",
            dropna: bool = True
        ) -> dict[str, list[NanPeriod]]:
        ...
    
    def find_nan_periods(self,
            station_number: list[str] | str | None = None,
            *,
            columns: list[str] | None = None,
            on_error: Literal["raise", "ignore"] = "raise",
            dropna: bool = True
        ) -> list[NanPeriod] | dict[str, list[NanPeriod]]:
        """
        Find all nan periods for the specified station(s). 

        Parameters
        ----------
        station_number : str | list[str] | None
            The station number(s) to filter by. If None, patterns for all stations will be returned.
        columns : list[str] | None, optional
            A list of column names to include in the analysis. If None, all columns will be used.
        on_error : Literal["raise", "ignore"] = "raise", optional
            How to handle the case where a station number is not found. If "raise", a ValueError will be raised. If "ignore", an empty list will be returned for that station.
        dropna : bool = True
            If True, rows with all NaN values (except for station number and datetime) will be dropped before analysis.
            
        Returns
        -------
        list[NanPeriod] | dict[str, list[NanPeriod]]
            A list of NanPeriod objects for the specified station number(s), or a dictionary mapping station numbers to lists of NanPeriod objects.
        
        Raises
        ------
        ValueError
            If the specified station number is not found in the DataFrame and on_error is set to "raise".
        """
        def identify_col_nan_segments(station_number: str, df: pd.DataFrame, column: str) -> Iterator[NanPeriod]:
            datetimes = df["datetime"].sort_values().reset_index(drop=True)
            new_segment = datetimes.diff().ne(pd.Timedelta(hours=1))
            segment_ids = new_segment.cumsum()
            for _, group in datetimes.groupby(segment_ids, sort=False):
                yield NanPeriod(
                    station_number,
                    column,
                    len(group),
                    group.iloc[0] - pd.Timedelta(hours=1),
                    group.iloc[-1],
                )
        # Main logic
        if on_error not in ["raise", "ignore"]:
            raise ValueError(f"Invalid value for on_error: {on_error}. Must be 'raise' or 'ignore'.")
        if station_number is None:
            return {stno: self.find_nan_periods(stno, columns=columns, on_error=on_error)
                    for stno in self.stations}
        elif isinstance(station_number, list):
            return {stno: self.find_nan_periods(stno, columns=columns, on_error=on_error)
                    for stno in station_number}
        try:
            station_data = self._processed_data(station_number, columns=columns, drop=dropna)
        except ValueError as e:
            if on_error == "raise":
                raise e
            else:
                return []
        return [
            period
            for column in self.__get_value_columns(station_data)
            for period in identify_col_nan_segments(
                station_number,
                station_data.loc[station_data[column].isna(), ["datetime", column]],
                column
            )
        ]