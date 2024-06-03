import random
import os
import shutil
def spliting_labels():
    random.seed(42)
    # Path to your images directory and labels file
    images_directory = '/home/yzhang63/deit-main/data'
    labels_file = '/data1/HDR+/2017/2017labels.txt'

    train_labels_file = '/data1/HDR+/2017/train_labels.txt'
    test_labels_file = '/data1/HDR+/2017/test_labels.txt'

    train_ratio = 0.8

    with open(labels_file, 'r') as f:
        lines = f.read().splitlines()

    random.shuffle(lines)

    split_idx = int(len(lines) * train_ratio)

    # Split the data
    train_data = lines[:split_idx]
    test_data = lines[split_idx:]

    # Save the training and test labels into separate files
    with open(train_labels_file, 'w') as f:
        f.write("\n".join(train_data))

    with open(test_labels_file, 'w') as f:
        f.write("\n".join(test_data))

    print(f"Training labels saved to {train_labels_file}")
    print(f"Test labels saved to {test_labels_file}")
    
def spliting_picture_data():
    datapath = "/data1/HDR+/2017/"
    new_type = ["train", "test"]
    file_type = ["dng","jpg"]
    
    for datatype in new_type:
        labelfile = os.path.join("/data1/HDR+/2017/", f"{datatype}_labels.txt")
    
        with open(labelfile, 'r') as f:
            lines = f.read().splitlines()
            
        for each_line in lines:
            for filetype in file_type:
                ori_path = os.path.join(datapath,filetype)
                new_path = os.path.join(datapath,datatype,filetype)
                
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                        
                src_file_name = each_line.split(".")[0] + '.' + filetype
                new_file = f"{new_path}.{filetype}"
                shutil.copy(os.path.join(ori_path, src_file_name), os.path.join(new_path))
                
spliting_labels()
spliting_picture_data()


