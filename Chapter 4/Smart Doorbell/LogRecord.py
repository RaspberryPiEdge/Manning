import csv
import os
import datetime

class LogRecord:
    def __init__(self, filepath):
        self.filepath = filepath

    def get_record(self):
        data = []
        with open(self.filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                datetime, name = row
                data.append({
                    'datetime': datetime,
                    'name': name
                })
        return data

    def record_timestamp(self, person):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([current_timestamp, person])
