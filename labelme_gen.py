import ImagePath as ImagePath
import os

data_path = 'json_file'

flle_list = ImagePath.read_image(data_path, 'json')

for file in flle_list:
    cmd = 'labelme_json_to_dataset ' + file
    os.system(cmd)