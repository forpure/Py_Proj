__author__ = 'zf'

import csv

rfile = open('../data/weibo_predict_data.txt', 'rb')
reader = csv.reader(rfile, 'excel-tab')
wfile = open('../data/simple_result.txt', 'wb')
writer=csv.writer(wfile, 'excel-tab')
usr_data = {}
for line in reader:
    writer.writerow([line[0], line[1], 0, 1, 2])

