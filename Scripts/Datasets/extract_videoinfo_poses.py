import glob
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import json 
import os
import shutil


def find_bbox(keypoints):
    keypoints_copy = keypoints
    to_delete = []
    cnt = 0
    for kp in keypoints_copy:
#         print(f"shape:{kp.shape}")
        pos = (kp[0], kp[1])
        if (pos == (0, 0)) or (pos == (1, 0)):
            to_delete.append(cnt)
        cnt += 1
    keypoints_copy = np.delete(keypoints_copy, to_delete, 0)
    return [min(keypoints_copy[:, 0]), max(keypoints_copy[:, 0]), min(keypoints_copy[:, 1]), max(keypoints_copy[:, 1])]


def get_player_skeletons(keypoints):
    all_bbox = []
    all_conf = []
    for kp in keypoints:
        all_bbox.append(find_bbox(kp))
        all_conf.append(sum(kp[:, 2]))

    all_bbox = np.array(all_bbox)
    all_conf = np.array(all_conf)
    filter_arr = all_conf < 10
    #     print(all_bbox)
    all_bbox_heights = all_bbox[:, 3] - all_bbox[:, 2]
    all_bbox_heights[filter_arr] = 0
    sortresult = list(all_bbox_heights.argsort())
    top2_bf = sortresult[-2:]
    if len(top2_bf) == 2:
        left_bf = top2_bf[0]
        right_bf = top2_bf[1]

        kp1 = keypoints[top2_bf[0]]
        kp2 = keypoints[top2_bf[1]]

        center_0 = sum(np.array(find_bbox(kp1))[0:2])
        center_1 = sum(np.array(find_bbox(kp2))[0:2])
        if center_0 > center_1:
            left_bf = top2_bf[1]
            right_bf = top2_bf[0]

        # Save pose_keypoints
        pose_keypoints_left = keypoints[left_bf]
        pose_keypoints_right = keypoints[right_bf]
        return [pose_keypoints_left, pose_keypoints_right]
    elif len(top2_bf) == 1:
        print(f"only 1 skeletons were found")
        return [keypoints[top2_bf[0]]]
    else:
        print(f"No skeletons were found")
        return []


def read_all_videos(path: str, file_format: str = "mp4") -> []:
    return glob.glob(path + "*." + file_format)


def get_video_index(filename: str) -> int:
    return int(filename.split('/')[-1].split('-')[-1][:-4])


def get_video_name(filename: str) -> str:
    return filename.split('/')[-1][:-4]


def get_crop(frame, bbox, target_dim: tuple):
    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], 0:3]
    crop = cv2.resize(crop, target_dim, interpolation=cv2.INTER_AREA)
    return crop

def mkdir(path):
    os.mkdir(path) if not os.path.exists(path) else print()

def extract_video_info_poses(video_path: str, file_format: str, json_path: str, mask_path: [], mask_bbox:[], export_path: str, dataset_identifier: str):
    mkdir(f"{export_path}")
    mkdir(f"{export_path}/{dataset_identifier}")
    mkdir(f"{export_path}/{dataset_identifier}/videos")
    mkdir(f"{export_path}/{dataset_identifier}/npys")
    mkdir(f"{export_path}/{dataset_identifier}/video_frames")

    mask_right = cv2.imread(mask_path[1], cv2.IMREAD_UNCHANGED)
    right_box = mask_bbox[1]
    crop_right_std = get_crop(mask_right, right_box, (16, 16))
    mask_left = cv2.imread(mask_path[0], cv2.IMREAD_UNCHANGED)
    left_box = mask_bbox[0]
    crop_left_std = get_crop(mask_left, left_box, (16, 16))

    videos = glob.glob(f"{video_path}/*.{file_format}")
    for v in videos:#[126:]:
        mkdir(f"{export_path}/{dataset_identifier}/video_frames/{get_video_index(v):05d}")
        cap = cv2.VideoCapture(v)
        frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decisive_frame_index = 0
        round_result = -1
        scores = []
        skeleton_frames = []

        print(f"{get_video_index(v):05d}\ {frame_counts}")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            # print(f"video: {get_video_index(filename)}, frame:{frame_cnt}")
            if ret:
                cv2.imwrite(f"{export_path}/{dataset_identifier}/video_frames/{get_video_index(v):05d}/{frame_count:05d}.jpg", frame)

                # Getting Skeletons
                # print(f"{json_path}/{get_video_index(v):05d}/{get_video_name(v)}_{frame_count:012}_keypoints.json")
                json_file = f"{json_path}/{get_video_index(v):05d}/{get_video_name(v)}_{frame_count:012}_keypoints.json"
                if os.path.exists(json_file):
                    with open(json_file, 'rb') as f:
                        jf = json.load(f)
                        all_skeletons = jf["people"]
                        keypoints = []
                        for sk in all_skeletons:
                            keypoints.append(np.array(sk['pose_keypoints_2d']).reshape(25, -1))                    
                        
                        for kp in keypoints:
                            kp[:, 0] = kp[:, 0] / frame.shape[1] # pos_x/width
                            kp[:, 1] = kp[:, 1] / frame.shape[0] # pos_y/height

                    player_skeletons = get_player_skeletons(keypoints)
                    if len(player_skeletons) != 2:
                        print(f"at frame #{frame_count}")
                else:
                    print(f"frame #{frame_count} need json file")
                    break

                # if len(player_skeletons) > 0:
                #     for sk in player_skeletons:
                #         sk[:, 0] = sk[:, 0] / frame.shape[1] # pos_x/width
                #         sk[:, 1] = sk[:, 1] / frame.shape[0] # pos_y/height
                skeleton_frames.append(player_skeletons)

                # Judging winner
                crop_left = get_crop(frame, left_box, (16, 16))
                crop_right = get_crop(frame, left_box, (16, 16))
                score_left = ssim(crop_left[:, :, 2], crop_left_std[:, :, 2])
                score_right = ssim(crop_right[:, :, 1], crop_right_std[:, :, 1])
                # print(f"score_left:{score_left} \n score_right:{score_right}")
                if round_result == -1:
                    if score_right > 0.9 or score_left > 0.9:
                        decisive_frame_index = frame_count
                        if score_right > 0.9 and score_left <= 0.9: # right win
                            round_result = 1
                        elif score_right <= 0.9 and score_left > 0.9:  # left win
                            round_result = 0
                        else: # both win
                            round_result = 2
                frame_count += 1
                scores.append([score_left, score_right])
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        # Export data to npy files
        scores = np.array(scores)
        # if skeleton_frames != []:
        #     skeleton_frames = np.stack(skeleton_frames, axis=0)
        export_file_name = f"{export_path}/{dataset_identifier}/npys/{get_video_index(v):05d}_info.npy"
        with open(export_file_name, 'wb') as f:
            np.savez(f,
                     frame_counts=frame_counts,
                     decisive_frame_index=decisive_frame_index,
                     round_result=round_result,
                     skeleton_frames=skeleton_frames,
                     scores=scores
                     )

        # copy & rename the video
        shutil.copy(v, f"{export_path}/{dataset_identifier}/videos/{get_video_index(v):05d}.{file_format}")

if __name__ == "__main__":
    extract_video_info_poses(video_path="../../Datasets/TokyoOlympics/video",
                             file_format='m4v',
                             json_path="../../Datasets/TokyoOlympics/openpose_results",
                             mask_path=["../../Datasets/TokyoOlympics/marker_crop/crop_1.png",
                                          "../../Datasets/TokyoOlympics/marker_crop/crop_2.png"],
                             mask_bbox=[[578, 655, 606, 682], [672, 655, 700, 682]],
                             export_path="../../Results/Datasets",
                             dataset_identifier="olympic_batch_2v4"
                             )
