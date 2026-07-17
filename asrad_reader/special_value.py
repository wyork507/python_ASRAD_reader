from dataclasses import dataclass
from enum import StrEnum
from unittest import case

class SpecialValue(StrEnum):
    """
    The special values used in observational data, along with their corresponding Big5 and UTF8 encodings.
    """
    WaitFix = "Instrument is waiting for repair."
    """儀器故障待修"""
    InBelow = "Data will be accumulated later."
    """資料將累計於後"""
    Trouble = "Data unavailable due to *instrument trouble*."
    """因 *故障(UTF8)* 而無資料"""
    Unknown = "Data unavailable due to *unknown reason* (or *instrument trouble*, only Big5)."
    """因 *不明原因(Big5/UTF8)* 或 *故障(Big5)* 而無資料"""
    OnTrace = "On trace."
    """雨跡(Trace)"""
    NoInObs = "Data unavailable due to no observation."
    """未觀測而無資料"""

    @property
    def big5(self) -> list[str]:
        """Returns the Big5 encoding list for the special value."""
        s = SpecialValue
        match self:
            case s.WaitFix:
                return ["-9991"]
            case s.InBelow:
                return ["-9996"]
            case s.Trouble | s.Unknown:
                return ["-9997"]
            case s.OnTrace:
                return ["-9998"]
            case s.NoInObs:
                return ["-9999"]
    
    @property
    def utf8(self) -> list[str]:
        """Returns the UTF8 encoding list for the special value."""
        s = SpecialValue
        match self:
            case s.WaitFix:
                return ["-999.1"]
            case s.InBelow:
                return ["-9.6", "-999.6"]
            case s.Trouble:
                return ["-9.5", "-99.5", "-999.5", "-9999.5"]
            case s.Unknown:
                return ["-9.7", "-99.7", "-999.7", "-9999.7"]
            case s.OnTrace:
                return ["-9.8"]
            case s.NoInObs:
                return ["None"]
    
    @property
    def numerical(self) -> list[float]:
        """Returns the numerical representation of the special value, if applicable."""
        s = SpecialValue
        together = self.big5
        if self is not s.NoInObs:
            together += self.utf8
        return [float(value) for value in together]
    
    @staticmethod
    def all_cases() -> set["SpecialValue"]:
        """Returns a set of all special values."""
        return {member for member in SpecialValue}