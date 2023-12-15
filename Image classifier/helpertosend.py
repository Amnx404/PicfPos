# %%
import cv2
import numpy as np
import os

def save_frame(cap, frame_idx, output_path, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(output_path, f'frame_{index + 1}.png'), frame)

def extract_frames_with_opencv(video_path, output_path, num_images):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
    if total_frames < num_images:
        frame_indices = np.round(np.linspace(0, total_frames - 1, num_images)).astype(int)
    else:
        frame_indices = np.sort(np.random.choice(np.arange(0, total_frames - 1), num_images, replace=False))
    existing_files = 0
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        existing_files = len([name for name in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, name))])
        num_images += existing_files

    for i, frame_idx in enumerate(frame_indices):
        save_frame(cap, frame_idx, output_path, i + existing_files)

    cap.release()

# Example usage:
# extract_frames_with_opencv('path/to/video.mp4', 'path/to/output/folder', 10)


# %% [markdown]
# # video Extractor from folder

# %%
import os
import glob
#list only directories
folders = [f for f in glob.glob(r'./input folder/mks_videos3/*') if os.path.isdir(f)]

# folder = './input folder/'
folders

# %%
# print(folders)
target_name = 'mks_videos3'
for folder in folders:
    videos = glob.glob(f'{folder}/*.mp4')
    print(videos)
    for video in videos:
        # from './input folder/anshika\\3d\\VID20231126114723.mp4 extract 3d
        extracted_folder = video.split('\\')[1]
        # path_to_save
        path_to_save = f'frames/{target_name}/{folder}/'
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        extract_frames_with_opencv(video, path_to_save, 400)

# %%


# %% [markdown]
# # Copy video from 1 folder to dataset folder

# %%
folder_path = r'C:\Users\aman.sa\Documents\code\PicfPos\Image classifier\frames\mks_videos3\input folder\mks_videos3'
folders = os.listdir(folder_path)
folders

# %%
for i in folders:
    files = os.listdir(f'{folder_path}/{i}')
    print(len(files))
    #increment if duplicate, copy each file to a new folder
    target = f'C:/Users/aman.sa/Documents/code/PicfPos/Image classifier/frames/dataset/{i}'
    target_symbol = 'mks_videos3'
    if not os.path.exists(target):
        os.makedirs(target)
    for j in files:
        #if files exists in target, increment and save to target
        count = 0
        while os.path.exists(f'{target}/{target_symbol}_{count}.png'):
            count += 1
        cv2.imwrite(f'{target}/{target_symbol}_{count}.png', cv2.imread(f'{folder_path}/{i}/{j}'))


# %%
folders

# %% [markdown]
# # Utility Scripts
# 

# %%
#copy all files from s5 to c5 and rename them
import os
import shutil

source = r'C:\Users\aman.sa\Documents\code\PicfPos\Image classifier\frames\dataset\c5'
target = r'C:\Users\aman.sa\Documents\code\PicfPos\Image classifier\frames\dataset\s5'

files = os.listdir(source)
for i in files:
    #increment if duplicate, copy each file to a new folder with the same name
    count = 0
    while os.path.exists(f'{target}/{i}_{count}.png'):
        count += 1
    shutil.copy(f'{source}/{i}', f'{target}/{i}_{count}.png')
    
    
    

# %%
#take a folder and then copy each file to a new folder with the same name
import os
import shutil

source = r'C:\Users\aman.sa\Documents\code\PicfPos\Image classifier\input folder\rec'

files = os.listdir(source)


for i in files:
    if not os.path.exists(f'{source}/{i.split(".")[0]}'):
        os.makedirs(f'{source}/{i.split(".")[0]}')
    shutil.copy(f'{source}/{i}', f'{source}/{i.split(".")[0]}/{i}')
    


