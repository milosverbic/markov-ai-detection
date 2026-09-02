import csv
from itertools import islice
import random as R

csv.field_size_limit(100 * 1024 * 1024) #100 mb row limit

target_row_index = R.randint(0,1431)  # 0-indexed row number you want to read
print(target_row_index, "\n", sep='')


with open("leto/enron_emails/susan.csv", mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    # Fast-forward directly to the target row without storing skipped rows in memory
    target_row = next(islice(reader, target_row_index, target_row_index + 1), None)

print(target_row[0] + "\n\n" + target_row[1])