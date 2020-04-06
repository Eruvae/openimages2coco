import csv
import numpy as np
import pycocotools.mask as mask
import os
import argparse
import imagesize
import base64
import shutil

parser = argparse.ArgumentParser(description='Convert open images segmentation data to coco format.')
parser.add_argument('--images', '-i', default='train_e', help='Folder that contains the images (as jpg)')
parser.add_argument('--classes', '-c', default='class-descriptions-boxable.csv', help='CSV file that contains class id mappings')
parser.add_argument('--annotations', '-a', default='oidv6-train-annotations-bbox.csv', help='CSV file that contains information about annotations')
parser.add_argument('--classes_out', '-co', default='openimages.names', help='Contains list of class names')
parser.add_argument('--images_out', '-io', default='images.list', help='Contains list of images')

args = parser.parse_args()

current_abs_dir = os.path.abspath(os.path.dirname(__file__))
image_path = args.images
image_path_abs = os.path.join(current_abs_dir, image_path)
label_path = os.path.join(args.images, 'labels')

if os.path.exists(label_path) and os.path.isdir(label_path):
    shutil.rmtree(label_path)

os.mkdir(label_path)

# Read class descriptions
print('Reading class descriptions')
class_list = []
class_id_map = {}
with open(args.classes, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    index = 0
    for row in reader:
        class_list.append(row[1])
        class_id_map[row[0]] = index
        index += 1
        
# Write class file
print('Writing class file')

with open(args.classes_out, 'w', newline='') as classfile:
    for class_name in class_list:
        classfile.write(class_name + '\n')
        
# Write image list file
image_list_file = open(args.images_out, 'w', newline='')

print('Counting lines')
with open(args.annotations, newline='') as file:
    row_count = sum(1 for line in file)
    print(row_count)

print('Reading annotations')
with open(args.annotations, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    index = 1
    last_image_name = None
    write_file = None
    writer = None
    
    for row in reader:
        line_count += 1
        if line_count % 10000 == 0:
            print(f'Bbox read progress: {(line_count / row_count * 100):.2f} %')
        
        if line_count == 1:
            print(f'Column count: {len(row)}')
            print(f'Column names are {", ".join(row)}')
        else:
            image_name = row[0]
            image_filepath = os.path.join(image_path, image_name + '.jpg')
            image_filepath_abs = os.path.join(image_path_abs, image_name + '.jpg')
            
            class_identifier = row[2]
            
            if class_identifier not in class_id_map:
                continue
            
            if not os.path.exists(image_filepath):
                continue
                
            if image_name != last_image_name:
                if write_file is not None:
                    write_file.close()
                
                image_list_file.write(image_filepath_abs + '\n')
                write_filepath = os.path.join(label_path, (image_name + '.txt'))
                write_file = open(write_filepath, 'w', newline='')
                writer = csv.writer(write_file, delimiter=' ')
                
                last_image_name = image_name
            
            class_id = class_id_map[class_identifier]
            xmin,xmax,ymin,ymax = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            xcent = (xmin + xmax) / 2
            ycent = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            
            writer.writerow([class_id, xcent, ycent, width, height])

if write_file is not None:
    write_file.close()
    
image_list_file.close()

print('Done')