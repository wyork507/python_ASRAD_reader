from pathlib import Path
from ASRAD_reader import NanMode
import ASRAD_reader as Reader

df = Reader.DataSet.read_file(Path("datas/20029999_cwb_hr/20021099.cwb_hr.txt"), mode = NanMode.AllValue)