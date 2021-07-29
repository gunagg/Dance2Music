import numpy as np
import argparse
import json
from PIL import Image
import os
from tqdm import tqdm

join=os.path.join

parser = argparse.ArgumentParser()
parser.add_argument('--input_json_dir', type=str, default='')
parser.add_argument('--output_feats_dir', type=str, default='')

args = parser.parse_args()


def load_data(json_dir, remove_hand_keypoints=True, remove_face_keypoints=True):
	print('---------- Loading pose keypoints ----------')
	
	pose = []
	fnames = []
	for json_fname in sorted(os.listdir(json_dir)):
		fnames.append(json_fname)
		json_fname = join(json_dir, json_fname)
		# print(json_fname)
		with open(json_fname) as f:
			keypoint_dicts = json.loads(f.read())['people']
			if len(keypoint_dicts) > 0:
				keypoint_dict = keypoint_dicts[0]
				pose_points = np.array(keypoint_dict['pose_keypoints_2d']).reshape(25, 3)[:, :-1].reshape(-1)
				face_points = np.array(keypoint_dict["face_keypoints_2d"]).reshape(70, 3)[:, :-1].reshape(-1) \
					if not remove_face_keypoints else []
				hand_points_l = np.array(keypoint_dict["hand_left_keypoints_2d"]).reshape(21, 3)[:, :-1].reshape(-1) \
					if not remove_hand_keypoints else []
				hand_points_r = np.array(keypoint_dict["hand_right_keypoints_2d"]).reshape(21, 3)[:, :-1].reshape(-1) \
					if not remove_hand_keypoints else []
				# print("hand points ", hand_points_r)
				key_points = np.concatenate([pose_points, face_points, hand_points_l, hand_points_r], 0)
				pose.append(key_points.tolist())

	return pose, fnames

def save_feats(pose, fname):
	pose = np.array(pose)
	np.save(fname, pose[:,:50])


inp_dir = args.input_json_dir
dirs = [join(inp_dir, dir_name) for dir_name in os.listdir(inp_dir)]
out_dir = args.output_feats_dir

if not os.path.exists(out_dir):
	os.makedirs(out_dir)

print('the total dirs are ', len(dirs))
for dir_name in tqdm(dirs):

	out_fname = join(out_dir, dir_name.split("/")[-1])
	pose, fnames = load_data(dir_name, remove_face_keypoints=True,
								 remove_hand_keypoints=True)
	save_feats(pose, fname = out_fname)




