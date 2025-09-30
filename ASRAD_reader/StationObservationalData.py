import pandas as pd
from ASRAD_reader import ObservationalDataset

class StationObservationalData(pd.DataFrame):
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
            return ObservationalDataset.transformation(self.value())
        
        @property
        def drop_nan(self) -> ObservationalDataset.StationData:
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
        
        def get_item_with_time(self, cols: list[str]) -> ObservationalDataset.StationData | None:
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