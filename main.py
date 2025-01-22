from pathlib import Path
import ASRAD_reader as Reader

a = Reader.DataSet.read_file(Path("E://github/summer/datas/20029999_cwb_hr/20021099.cwb_hr.txt"), mode = Reader.NanMode.AllValue)
#a = Reader.DataSet.read_dataset_csv("data.csv")

b = a.daily_average
c = a.mouth_average


print("YOLO")