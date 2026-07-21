from __future__ import annotations
from typing import List
from enum import Enum

from .special_value import SpecialValue


class NanMode(Enum):
    """
    The enumeration that defines different modes for handling NaN values in observational data.
    """
    _value_: List[SpecialValue]
    __s = SpecialValue

    ObsEmpty = [__s.NoInObs, __s.WaitFix, __s.Unknown, __s.Trouble]
    """無資料"""
    AllEmpty = [member for member in SpecialValue]
    """所有非數值"""

    @property
    def big5(self) -> List[str]:
        """Returns the Big5 encoding list for all special values in the current NanMode."""
        return [value for member in self.value for value in member.big5]
    
    @property
    def utf8(self) -> List[str]:
        """Returns the UTF8 encoding list for all special values in the current NanMode."""
        return [value for member in self.value for value in member.utf8]
    
    def __str__(self) -> str:
        cases = ',\n'.join([f'{member.value}' for member in self.value])
        f_strings = f"NanMode.{self.name}, which includes the following special values: \n{cases}"
        return f_strings

    def __repr__(self) -> str:
        return f"NanMode.{self.name}"