from dataclasses import dataclass
from enum import Enum

@dataclass(frozen = True)
class Encoding:
    """
    - Do not create Encoding instance directly, use `NanMode` instead. \n
    
    Attributes:
        Big5 (list[str]): List of special values in Big5 encoding.
        UTF8 (list[str]): List of special values in UTF8 encoding.
    """
    Big5: list[str]
    UTF8: list[str]

class SpecialValue(Enum):
    """
    - Do not use SpecialValue directly, use `NanMode` instead. \n

    |Cases    | Description           |
    |:--------|:----------------------|
    |`WaitFix`|儀器故障待修            |
    |`InBelow`|資料累計於後            |
    |`Trouble`|因 *故障(UTF8)* 而無資料|
    |`Unknown`|因 *不明原因(Big5/UTF8)*|
    |         |或 *故障(Big5)* 而無資料|
    |`OnTrace`|雨跡(Trace)            |
    |`NoInObs`|未觀測而無資料          |
    """

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

