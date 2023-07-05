import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from util import *

with open('/home/famousdeer/Desktop/Praca magisterska/Program/data/VOCdevkit/TEST_images.json', 'r') as j:
    image_file = json.load(j)

with open('/home/famousdeer/Desktop/Praca magisterska/Program/data/VOCdevkit/TEST_objects.json', 'r') as j:
    object_file = json.load(j) 
ROOT1 = image_file[1]
ROOT2 = image_file[15]

object1 = object_file[1]
object2 = object_file[15]

image = Image.open(ROOT2, mode='r')
image = image.convert('RGB')

draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
print(object2['labels'])
det_boxes = len(object2['boxes'])

for i in range(det_boxes):
    draw.rectangle(xy=object2['boxes'][i], outline=label_color_map['car'])
    text_size = font.getsize('CAR')
    text_location = [object2['boxes'][i][0] + 2., object2['boxes'][i][1] - text_size[1]]
    textbox_location = [object2['boxes'][i][0], 
                        object2['boxes'][i][1] - text_size[1], 
                        object2['boxes'][i][0] + text_size[0] + 4.,
                        object2['boxes'][i][1]]
    draw.rectangle(xy=textbox_location, fill=label_color_map['car'])
    draw.text(xy=text_location, text='CAR', fill='white',font=font)
image.show()