import csv
import numpy as np
import pycocotools.mask as mask
import os
import argparse
import imagesize
import base64
import shutil

parser = argparse.ArgumentParser(description='Convert open images segmentation data to coco format.')
parser.add_argument('--annotations', '-a', default='oidv6-train-annotations-bbox.csv', help='CSV file that contains information about annotations')
parser.add_argument('--filter', '-f', default='e', help='Filter set (1 - F)')
parser.add_argument('--annotations_out', '-ao', default='oidv6-train-annotations-bbox-e.csv', help='Filtered annotations')

args = parser.parse_args()

print('Counting lines')
with open(args.annotations, newline='') as file:
    row_count = sum(1 for line in file)
    print(row_count)

print('Reading annotations')
with open(args.annotations, newline='') as infile:
    outfile = open(args.annotations_out, 'w', newline='')
    reader = csv.reader(infile, delimiter=',')
    writer = csv.writer(outfile, delimiter=',')
    line_count = 0
    
    for row in reader:
        line_count += 1
        if line_count % 10000 == 0:
            print(f'Filter progress: {(line_count / row_count * 100):.2f} %')
        
        if line_count == 1:
            print(f'Column count: {len(row)}')
            print(f'Column names are {", ".join(row)}')
            writer.writerow(row)
        else:
            image_name = row[0]
            if (image_name[0] == args.filter):
                writer.writerow(row)

outfile.close()

print('Done')