import numpy as np
import pandas as pd
import os
from tqdm import tqdm

root = '../collections/'
people = os.listdir(root)
training_list = []
separators = pd.read_csv('separator.csv', header=None).values
for person in tqdm(people):
    person_path = os.path.join(root, person)
    fingers = os.listdir(person_path)
    for finger in fingers:
        finger_path = os.path.join(person_path, finger)
        poses = os.listdir(finger_path)
        for pose in poses:
            pose_path = os.path.join(finger_path, pose)
            force_file = os.path.join(pose_path, 'force.txt')
            data = np.loadtxt(force_file, skiprows=1)

            end = -1
            for separator in separators:
                if separator[0] == person and separator[1] == int(finger) and separator[2] == int(pose):
                    end = separator[3]
                    break

            if end == -1:
                continue

            for i in range(end):
            # for i in range(end, data.shape[0]):

                filename, force_x, force_y, force_z, touch_x, touch_y = data[i]
                if touch_x == 0 or touch_y == 0 or force_z <= 20 or force_z > 1000:
                    continue
                for j in range(i+2,i+100,2):
                    if j >= end:
                    # if j >= data.shape[0]:
                        continue
                    filename, force_x, force_y, force_z, touch_x, touch_y = data[i]
                    filename2, force_x2, force_y2, force_z2, touch_x2, touch_y2 = data[j]
                    if touch_x2 == 0 or touch_y2 == 0 or force_z2 > 1000 or force_z2 <= 20:
                        continue
                    filename = str(int(filename/10)) + '_0.npy'
                    filename2 = str(int(filename2 / 10)) + '_0.npy'
                    training_list.append([pose_path, filename, filename2, force_x2 - force_x, force_y2 - force_y, force_z2 - force_z, touch_x, touch_y, touch_x2, touch_y2, 0])

training_list = pd.DataFrame(training_list)
training_list.to_csv('training_lists2/train_list_press.csv', index=False)
# training_list.to_csv('training_lists2/train_list_shear.csv', index=False)




