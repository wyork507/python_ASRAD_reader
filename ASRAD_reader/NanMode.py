from enum import Enum
from ASRAD_reader.SpecialValue import SpecialValue, Encoding

class NanMode(Enum):
    """
    A enumeration class, with 4 mode, or you can create a custom Encoding instance.
    |Modes   | Description                                          |
    |:-------|:-----------------------------------------------------|
    |`ObsEmpty`|Encoding instance for any reason without observation. |
    |`AllEmpty`|Encoding instance for all reasons except trace.       |
    |`NotInObs`|Encoding instance for no data due to no observation.  |
    |`AllValue`|Encoding instance for all special values.             |
    |`Custom()`|Return a custom Encoding instance for your parameters.|
    """

    @staticmethod
    def _merge(*values: SpecialValue.value) -> Encoding:
        """
        - Do not use _merge directly, use `Custom` instead. \n
        
        Merges multiple SpecialValue instances into a single Encoding instance.
        Args:
            *values: Variable length SpecialValue instances to merge.
        Returns:
            Encoding: A new Encoding instance containing merged Big5 and UTF8 lists.
        """
        big5 = []
        utf8 = []
        for val in values:
            big5 += val.Big5 or []
            utf8 += val.UTF8 or []
        return Encoding(big5, utf8)
    
    @staticmethod
    def Custom(WaitFix: bool = False,
               InBelow: bool = False,
               Trouble: bool = False,
               Unknown: bool = False,
               OnTrace: bool = False,
               NoInObs: bool = False) -> Encoding:
        """
        Creates a custom Encoding instance with specified special values.
        Args:
            WaitFix (bool): Whether to include WaitFix special value.
            InBelow (bool): Whether to include InBelow special value.
            Trouble (bool): Whether to include Trouble special value.
            Unknown (bool): Whether to include Unknown special value.
            OnTrace (bool): Whether to include OnTrace special value.
            NoInObs (bool): Whether to include NoInObs special value.
        Returns:
            Encoding: A new Encoding instance with the provided special values.
        """
        return NanMode._merge(*[
            SpecialValue.WaitFix.value if WaitFix else None,
            SpecialValue.InBelow.value if InBelow else None,
            SpecialValue.Trouble.value if Trouble else None,
            SpecialValue.Unknown.value if Unknown else None,
            SpecialValue.OnTrace.value if OnTrace else None,
            SpecialValue.NoInObs.value if NoInObs else None,
        ])


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

