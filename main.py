from pathlib import Path
from ASRAD_reader import NanMode
from ASRAD_reader.loader import load_file

df = load_file(Path("datas/20029999_cwb_hr/20021099.cwb_hr.txt"), mode=NanMode.AllValue)