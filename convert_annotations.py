import csv
import json
import cv2
import numpy as np
import pycocotools.mask as mask
import os
import argparse
import imagesize
import base64

parser = argparse.ArgumentParser(description='Convert open images segmentation data to coco format.')
parser.add_argument('--images', '-i', default='train_e', help='Folder that contains the images (as jpg)')
parser.add_argument('--masks', '-m', default='train-masks-e', help='Folder that contains the masks (as png)')
parser.add_argument('--classes', '-c', default='class-descriptions-boxable.csv', help='CSV file that contains class id mappings')
parser.add_argument('--annotations', '-a', default='train-annotations-object-segmentation.csv', help='CSV file that contains information about annotations')
parser.add_argument('--remove_unknown_masks', action='store_true', help='Masks from classes not specified in the class id mappings are removed from the mask folder')
parser.add_argument('--generate_yolact_config', action='store_true', help='Generates json file with information needed for yolact custom dataset class')

args = parser.parse_args()

image_path = args.images
mask_path = args.masks

# Setup basic json structure

data = {
    'info': {},
    'licenses': [{
        "url": "http://creativecommons.org/licenses/by/2.0/",
        "id": 0,
        "name": "Attribution License"
    }],
    'images': [],
    'annotations': [],
    'categories': []
}

info = data['info']
info['description'] = 'Open Images Dataset'
info['url'] = 'https://storage.googleapis.com/openimages/web/index.html'
info['version'] = 6
info['year'] = 2020
info['contributor'] = 'Open Images Project Group'
info['date_created'] = '2020/02'

# Read class descriptions
print('Reading class descriptions')
class_list = ['BG']
class_id_map = {}
with open(args.classes, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    index = 1
    for row in reader:
        class_list.append(row[1])
        class_id_map[row[0]] = index
        
        category = {}
        category['id'] = index
        category['name'] = row[1]
        data['categories'].append(category)
        
        index += 1

# Generate yolact config
if args.generate_yolact_config:
    print('Generating yolact config')
    yolact_data = {}
    yolact_data['name'] = 'OpenImages Dataset'
    yolact_data['train_images'] = image_path
    yolact_data['train_info'] = 'coco_annotations.json'
    yolact_data['valid_images'] = image_path
    yolact_data['valid_info'] = 'coco_annotations.json'
    yolact_data['has_gt'] = True
    yolact_data['class_names'] = class_list[1:]

    with open('yolact_config.json', 'w') as outfile:
        json.dump(yolact_data, outfile)

# Read images
print('Reading images')
#MAX_IMAGE_COUNT = 1000
imgdata_map = {}
directory = os.fsencode(image_path)
index = 1
files = os.listdir(directory)
num_imgs = len(files)
print('Files', num_imgs)
for file in files:
    filename = os.fsdecode(file)
    filepath = os.path.join(image_path, filename)
    if filename.endswith('.jpg'):    
        width, height = imagesize.get(filepath)
        imgdata_map[filename] = {'id': index, 'width': width, 'height': height}
        image = {}
        image['file_name'] = filename
        image['height'] = height
        image['width'] = width
        image['id'] = index
        data['images'].append(image)
        
        index += 1
        #if index > MAX_IMAGE_COUNT:
        #    break
        
        if index % 1000 == 0:
            print(f'Img read progress: {(index / num_imgs * 100):.2f} %')
    

# Go through all annotations, add if mask found

#MAX_MASK_COUNT = 1000

print('Counting lines')
with open(args.annotations, newline='') as file:
    row_count = sum(1 for line in file)
    print(row_count)

print('Reading masks')
with open(args.annotations, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    index = 1
    
    for row in reader:
        line_count += 1
        if line_count % 10000 == 0:
            print(f'Mask read progress: {(line_count / row_count * 100):.2f} %')
        
        if line_count == 1:
            print(f'Column count: {len(row)}')
            print(f'Column names are {", ".join(row)}')
        else:
            file_name = row[0]
            filepath = os.path.join(mask_path, file_name)
            image_name = row[1] + '.jpg'
            class_identifier = row[2]
            
            if class_identifier not in class_id_map:
                if args.remove_unknown_masks and os.path.exists(filepath):
                    os.remove(filepath)
                continue
            
            if image_name not in imgdata_map:
                #print(f'Image {image_name} for mask {file_name} not found')
                continue
                
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'Warning: Mask {file_name} not found')
                continue
             
            imgdata = imgdata_map[image_name]
            image_id = imgdata['id']

            category_id = class_id_map[class_identifier]

            # Rescale masks to be same size as associated images, and binarize masks
            img = cv2.resize(img,(imgdata['width'],imgdata['height']))
            img = img.clip(max=1)
            
            # RLE segmentation
            m = mask.encode(np.asfortranarray(img))
            area = int(mask.area(m))
            x, y, w, h = mask.toBbox(m)
            
            """Encode compressed RLEs with base64 to be able to store in JSON
            Note: this is not COCO standard
            If you use pycocotools, use the following function
            to convert the imported annotations back to bytes:
            
            from pycocotools.coco import COCO
            import base64
            
            def decode_base64_rles(coco):
                for ann in coco.dataset['annotations']:
                    segm = ann['segmentation']
                    if type(segm) != list and type(segm['counts']) != list:
                        segm['counts'] = base64.b64decode(segm['counts'])
            
            coco = COCO('annotations.json')
            decode_base64_rles(coco)
            """
            #print('Original', m['counts'], '\n')
            m['counts'] = base64.b64encode(m['counts']).decode('utf-8')
            #print('Encoded', m['counts'], '\n')
            #mdec = base64.b64decode(m['counts'])
            #print('Decoded', mdec, '\n')
            
            # Opencv contour segmentation (not used)
            #contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #area = cv2.contourArea(contours[0])
            #x,y,w,h = cv2.boundingRect(contours[0])
            #dbg_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            #dbg_img = cv2.drawContours(dbg_img, contours, -1, (0,0,255), thickness=2)
            #cv2.imshow('image',dbg_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            annotation = {}
            annotation['segmentation'] = m #contours
            annotation['area'] = area
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [x, y, w, h]
            annotation['category_id'] = category_id
            annotation['id'] = index
            
            data['annotations'].append(annotation)
            
            index += 1
            
        #if index > MAX_MASK_COUNT:
        #    break
    
print('Input data processed, writing json')
    
with open('coco_annotations.json', 'w') as outfile:
    json.dump(data, outfile)
    
print('Done')