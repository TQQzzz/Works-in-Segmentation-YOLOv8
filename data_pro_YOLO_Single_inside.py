import json
import os
import yaml
import shutil
import numpy as np
from PIL import Image
from pathlib import Path


def divide_files_into_folder(main_folder,row_folder):
    """
    Move image and JSON files from a main folder to separate image and JSON folders.

    Args:
        main_folder (str): The directory containing the unorganized files.
        row_folder (str): The directory to which the organized files will be moved.
    """
    output_folder_images = row_folder
    output_folder_jsons = row_folder
    # Iterate through the subfolders in the main folder
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        # Check if the path is a folder and not an output folder
        if os.path.isdir(subfolder_path) and subfolder != output_folder_images and subfolder != output_folder_jsons:
            
            # Iterate through the files in the subfolder
            for file in os.listdir(subfolder_path):
                if file.startswith("._"):
                    # Delete the file
                    file_path = os.path.join(subfolder_path, file)
                    os.remove(file_path)
                    continue

                file_path = os.path.join(subfolder_path, file)

                # Check if the file is an image (assuming .jpg or .png format)
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Check if the filename contains the word "Building"
                    #if 'diaphragm' not in file.lower() and 'structural' not in file.lower():
                    if "diaphragm" in file.lower():
                    #if 1:
                        # Copy the image to the images folder
                        shutil.copy(file_path, output_folder_images)

                # Check if the file is a .json file
                elif file.lower().endswith(".json"):
                    # Check if the filename contains the word "Building"
                    if "diaphragm" in file.lower():
                    #if 1:
                        # Copy the .json file to the json_files folder
                        shutil.copy(file_path, output_folder_jsons)

    print("Files have been successfully divided into images and json_files folders within the data folder.")


def clean_label(label):
    '''
    Cleans a label by removing trailing 's' and converting it to uppercase.

    Args:
        label (str): The label to clean.

    Returns:
        str: The cleaned label.
    '''
    # Remove trailing 's'
    if label.endswith('s'):
        label = label[:-1]

    # Convert to uppercase
    label = label.upper()

    return label

def change_label_name(label):
    #this part is to transform the old names to new ones.
    label_dict = {
        "Window": "Window",
        "Precast-RC-slabs": "RC-Slab",
        "RC-solid-slab": "RC-Slab",
        "RC-Joist": "RC-Joist",
        "PC1": "PC",
        "PC2": "PC",
        "RC-Column": "RC-Columns",#inside
        "Slab": "RC2-Slab",
        "UCM/URM7": "Brick",
        "Timber-Frame": "Timber",
        "Timber-Column": "Timber-Column",
        "Timber-Joist": "Timber-Joist",
        "Light-roof": "Light-roof",
        "UCM/URM4": "Brick",
        "RM1": "Masonry",
        "Adobe": "Brick",
        #"RC2": "RC",
        'RC2-Column':'RC2-Column'#outside
    }
    
    return label_dict.get(label, label)


def clean_labels(data_folder):
    '''
    Cleans and transfer the labels in JSON files within the specified data folder. 

    Args:
        data_folder (str): The path to the data folder.
    '''
    for file in os.listdir(data_folder):
        if file.lower().endswith(".json"):
            # Load the .json file
            with open(os.path.join(data_folder, file), 'r') as json_file:
                data = json.load(json_file)
                new_shapes = []

                for obj in data['shapes']:
                    #print(obj['label'])

                    new_label = change_label_name(obj['label'])  

                    #print(new_label)

                    # clean the label
                    new_label = clean_label(new_label)
                    #(new_label)

                    if new_label in class_map:
                        obj['label'] = new_label
                        print(new_label)
                        new_shapes.append(obj)
                    
                    #print("----------------------------")

                # Replace the old shapes with the new ones
                data['shapes'] = new_shapes

            # Save the cleaned json file
            with open(os.path.join(data_folder, file), 'w') as json_file:
                json.dump(data, json_file)
    print('clean successfully')


def transform_annotations(data_folder, output_folder):
    """
    Transform the annotations from JSON files into a format suitable for segmentation tasks.

    Args:
        data_folder (str): The path to the directory that contains the original JSON files.
        output_folder (str): The path to the directory where the transformed files will be saved.
    """
    output_folder_images = os.path.join(output_folder, 'images')
    output_folder_labels = os.path.join(output_folder, 'labels')
    
    # Create the necessary folders if they do not exist
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_labels, exist_ok=True)

    # Create train and val folders for images and labels
    os.makedirs(os.path.join(output_folder_images, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_images, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_labels, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_labels, 'val'), exist_ok=True)

    # Split into train and val
    files = os.listdir(data_folder)
    np.random.shuffle(files)
    num_train = int(0.8 * len(files))
    train_files = files[:num_train]
    val_files = files[num_train:]

    # Function to handle each file
    def handle_file(file, output_folder_images, output_folder_labels):
        """
        Handle each file in the dataset. It extracts information from the JSON files,
        writes it to text files in the appropriate format, and moves the corresponding image
        to the designated output folder.

        Args:
            file (str): The filename of the JSON file to be processed.
            output_folder_images (str): The directory path where the images will be stored.
            output_folder_labels (str): The directory path where the labels (annotations) will be stored.
        """
        if file.lower().endswith(".json"):
            # Load the .json file
            with open(os.path.join(data_folder, file), 'r') as json_file:
                data = json.load(json_file)
                img_width = int(data['imageWidth'])
                img_height = int(data['imageHeight'])
                # Get the image file path
                image_file_name = Path(file).stem + ".jpg"
                image_file_path = os.path.join(data_folder, image_file_name)


                # Create a new .txt file for each .json file
                with open(os.path.join(output_folder_labels, f"{Path(file).stem}.txt"), 'w') as txt_file:
                    # Iterate through the objects in the .json file
                    for obj in data['shapes']:

                        # Check if label exists in the class_map
                        if obj['label'] not in class_map:
                            print(f"Skipping label not found in class_map: {obj['label']}")
                            continue
                        
                        points = np.array(obj['points'])

                        # Write object's class index and all boundary points to the txt file
                        class_id = class_map[obj['label']]
                        txt_file.write(f"{class_id}")
                        for point in points:
                            x, y = point
                            
                            # normalize coordinates to be between 0 and 1
                            x_normalized = x / img_width
                            y_normalized = y / img_height
                            txt_file.write(f" {x_normalized} {y_normalized}")
                            if(x_normalized>1 or x_normalized<0 or y_normalized>1 or y_normalized<0):
                                print(x_normalized,'-----',y_normalized)
                        txt_file.write("\n")
                print(image_file_path)
                # Copy the image to the new folder
                shutil.copyfile(image_file_path, os.path.join(output_folder_images, f"{Path(file).stem}.jpg"))

    # Process all the files
    for file in train_files:
        print(file)
        handle_file(file, os.path.join(output_folder_images, 'train'), os.path.join(output_folder_labels, 'train'))

    for file in val_files:
        handle_file(file, os.path.join(output_folder_images, 'val'), os.path.join(output_folder_labels, 'val'))

    # Write YAML file
    data_yaml = {
        'train': './images/train',
        'val': './images/val',
        'nc': len(class_map),
        'names': list(class_map.keys()),
    }
    
    #This is to write the yolo running comment 
    comment = f"#yolo segment train data=/scratch/tz2518/Segmentation_YOLO/{class_name}_data_YOLO_Single/data.yaml model=yolov8x-seg.yaml pretrained=/scratch/tz2518/ultralytics/yolov8x-seg.pt epochs=1000 imgsz=1024 cache=True name={class_name}"

    with open(os.path.join(output_folder, 'data.yaml'), 'w') as yaml_file:
        yaml_file.write(comment + "\n")
        yaml.dump(data_yaml, yaml_file, sort_keys=False)



# define the classes to save
all_classes = {
    "RC-SLAB": 0,
    "RC-JOIST": 1,
    "RC-COLUMN": 2,
    "TIMBER-COLUMN": 3,
    "TIMBER-JOIST": 4
}

# input and output path
main_folder = '/scratch/tz2518/data_7.3'
output_base_folder = '/scratch/tz2518/Segmentation_YOLO'


class_map = {}

# for every class
for class_name in all_classes.keys():

    # clean class_map and add one new class
    class_map.clear()
    class_map[class_name] = 0
    
    # create ouput folder
    output_folder = os.path.join(output_base_folder, f'{class_name}_data_YOLO_Single')
    os.makedirs(output_folder, exist_ok=True)
    # create folders in the ouput folder
    row_folder = os.path.join(output_folder, 'row')
    os.makedirs(row_folder, exist_ok=True)
    row_image = os.path.join(row_folder, 'image')
    os.makedirs(row_image, exist_ok=True)
    row_json = os.path.join(row_folder, 'json')
    os.makedirs(row_json, exist_ok=True)

    # divide the images to one folder
    divide_files_into_folder(main_folder, row_folder)
    #clean the json files
    clean_labels(row_folder)
    # transfer labels to yolo's
    transform_annotations(row_folder, output_folder)
