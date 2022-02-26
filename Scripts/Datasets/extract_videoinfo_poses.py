import glob
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import json
import os
import shutil

from FencingAnalysis.BodyframeHelper import get_player_skeletons


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
                    player_skeletons = get_player_skeletons(all_skeletons)
                    if len(player_skeletons) != 2:
                        print(f"at frame #{frame_count}")
                else:
                    print(f"frame #{frame_count} need json file")
                    break

                if len(player_skeletons) > 0:
                    for sk in player_skeletons:
                        sk[:, 0] = sk[:, 0] / frame.shape[1] # pos_x/width
                        sk[:, 1] = sk[:, 1] / frame.shape[0] # pos_y/height
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
    extract_video_info_poses(video_path="/media/weihao/UNTITLED/tokyo_olymics/videos2",
                             file_format='m4v',
                             json_path="/media/weihao/UNTITLED/tokyo_olymics/output_jsons2",
                             mask_path=["/media/weihao/UNTITLED/tokyo_olymics/marker_crop/crop_1.png",
                                          "/media/weihao/UNTITLED/tokyo_olymics/marker_crop/crop_2.png"],
                             mask_bbox=[[578, 655, 606, 682], [672, 655, 700, 682]],
                             export_path="/media/weihao/UNTITLED/tokyo_olymics/datasets",
                             dataset_identifier="olympic_batch_2v3"
                             )
