import os
from openpyxl import load_workbook

directory = os.getcwd() + "//" + "AddedFeaturesData//"


landmarks = []
names = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    wb = load_workbook(filename=directory + filename)
    ws = wb["Data"]
    ws.insert_cols(4)
    ws.move_range("AA2:AA1000", rows=0, cols=-23)
    ws.delete_cols(27)
    wb.save(directory + filename)

