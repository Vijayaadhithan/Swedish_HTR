import os
import shutil

source_dir = "/mimer/NOBACKUP/groups/snic2022-23-105/svenska_htr/Labours_Memo"
destination_dir = "/mimer/NOBACKUP/groups/snic2022-23-105/svenska_htr/train_all_LM"

def copy_xml_files(source_dir, destination_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png") or file.endswith(".JPG"):
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_dir, file)
                shutil.copy(source_file_path, destination_file_path)

copy_xml_files(source_dir, destination_dir)
