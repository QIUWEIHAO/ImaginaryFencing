import cv2
import glob
import numpy as np
from extract_videoinfo_poses import mkdir

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

def convert_to_screenspace(pose, screen_dims):
    pose_converted = np.zeros_like(pose)
    pose_converted[:, 0] = pose[:, 0] * screen_dims[0]
    pose_converted[:, 1] = pose[:, 1] * screen_dims[1]
    if pose.shape[1] == 3:
        pose_converted[:, 2] = pose[:, 2]
    return pose_converted

# pose [0-1] range
def find_pose_centers(pose):
    pose_centered = np.zeros_like(pose)
    pose_centered[:, 0] = pose[:, 0]
    pose_centered[:, 1] = pose[:, 1]
    if pose.shape[1] == 3:
        pose_centered[:, 2] = pose[:, 2]
    bbox = find_bbox(pose_centered)
    center = [(bbox[0] + bbox[1]) / 2.0, (bbox[2] + bbox[3]) / 2.0]
    pose_centered[:, 0] = pose_centered[:, 0] - center[0]
    pose_centered[:, 1] = pose_centered[:, 1] - center[1]
    return center, pose_centered, bbox

def get_pose_centered_crop(frame, pose, output_size:tuple = (256,256)):
    pose_copy = convert_to_screenspace(pose, screen_dims=[frame.shape[1], frame.shape[0]])
    center, pose_centered, bbox = find_pose_centers(pose_copy)
    center = [int(c) for c in center]
    bbox = [int(b) for b in bbox]
    crop_size = int(max(bbox[1] - bbox[0], bbox[3] - bbox[2]) * 0.8) * 2  # keep the crop size even number
    crop = np.zeros(shape=[crop_size, crop_size, 3], dtype=np.uint8)

    frame_bbox = [
        int(max(0, center[0] - crop_size / 2)),
        int(min(frame.shape[1], center[0] + crop_size / 2)),
        int(max(0, center[1] - crop_size / 2)),
        int(min(frame.shape[0], center[1] + crop_size / 2))
    ]
    offseted_bbox = [
        int(max(frame.shape[1], center[0] + crop_size / 2)) - frame.shape[1],
        crop_size - int(-min(0, center[0] - crop_size / 2)),
        int(max(frame.shape[0], center[1] + crop_size / 2)) - frame.shape[0],
        crop_size - int(-min(0, center[1] - crop_size / 2))
    ]
    # print(center)
    # print(bbox)
    # print(crop_size)
    # print(frame_bbox)
    # print(offseted_bbox)
    crop[offseted_bbox[2]:offseted_bbox[3], offseted_bbox[0]:offseted_bbox[1], :] = \
        frame[frame_bbox[2]:frame_bbox[3], frame_bbox[0]:frame_bbox[1], :]
    crop = cv2.resize(crop, dsize=output_size, interpolation=cv2.INTER_AREA)
    pose_centered[:, 0:2] = pose_centered[:, 0:2] / crop_size
    return crop, bbox, frame_bbox, pose_centered

def crop_pose_image(database_path: str, export_path: str, options: [], window_offsets:[] = [-5,5]):
    pose_files = glob.glob(f"{database_path}/npys/*")
    mkdir(f"{export_path}")
    mkdir(f"{export_path}/image_crops/")
    mkdir(f"{export_path}/pose_crop_info/")
    for op in options:
        if op == "all": # un-reverted
            mkdir(f"{export_path}/image_crops/all/")
            crop_id = 0
            crop_info = []
            crop_origin = []
            all_poses_centered = []
            for pf in pose_files[0:50]:
                print(pf)
                video_index = int(pf.split('_')[-2].split('/')[-1])
                pf_data = np.load(pf, allow_pickle=True)
                skeleton_frames = pf_data["skeleton_frames"]
                frame_count = 0
                for skf in skeleton_frames:
                    frame = cv2.imread(f"{database_path}/video_frames/{video_index:05d}/{frame_count:05d}.jpg" )
                    cnt = 0
                    for sk in skf:
                        crop, bbox, frame_bbox, pose_centered = get_pose_centered_crop(frame, sk)
                        # print(crop.shape)
                        # cv2.imwrite(f"{export_path}/image_crops/all/{crop_id:05d}_v_{video_index:05d}_f_{frame_count:05d}_{cnt}.jpg", crop)
                        cnt += 1
                        crop_id += 1
                        crop_info.append(np.stack([bbox, frame_bbox], axis=0)) # 2x4
                        crop_origin.append([crop_id, video_index, frame_count])
                        all_poses_centered.append(pose_centered) # 25 x 3
                    frame_count += 1

            crop_info = np.stack(crop_info, axis=0) # N x 2 x 4
            all_poses_centered = np.stack(all_poses_centered, axis=0) # N x 25 x 3
            # crop_origin = np.stack(crop_origin, axis=0) # N x 25 x 3
            export_file_name = f"{export_path}/pose_crop_info/all_poses_centered_info.npy"
            with open(export_file_name, 'wb') as f:
                np.savez(f,
                         all_poses_centered=all_poses_centered,
                         crop_info=crop_info,
                         crop_origin=crop_origin
                         )
        elif op == "pairs":
            mkdir(f"{export_path}/image_crops/pairs")

            pair_id = 0
            all_pose_pairs_per_video = []
            crop_info_per_video = []
            for pf in pose_files:
                video_index = int(pf.split('_')[-2].split('/')[-1])
                print(pf)
                pf_data = np.load(pf, allow_pickle=True)
                result = pf_data["round_result"]
                decisive_frame = pf_data["decisive_frame_index"]
                skeleton_frames = pf_data["skeleton_frames"]
                if result == -1 or result == 2 or len(skeleton_frames) < (decisive_frame + window_offsets[1]):
                    continue
                else:
                    crop_info = []
                    all_pose_pairs = []
                    for frame_count in range(max(0, decisive_frame + window_offsets[0]), decisive_frame + window_offsets[1]):
                        frame = cv2.imread(f"{database_path}/video_frames/{video_index:05d}/{frame_count:05d}.jpg")
                        skf = skeleton_frames[frame_count]
                        if(len(skf) != 2):
                            print(f"frame #{frame_count}: only {len(skf)} skeletons")
                            break
                        else:
                            winner_crop, winner_bbox, winner_frame_bbox, winner_pose_centered = get_pose_centered_crop(
                                frame, skf[result])
                            loser_crop, loser_bbox, loser_frame_bbox, loser_pose_centered = get_pose_centered_crop(
                                frame, skf[1-result])

                            flip_flag = -1
                            if result == 1:
                                winner_pose_centered[:, 0] = - winner_pose_centered[:, 0]
                                winner_crop = cv2.flip(winner_crop, 1)
                                flip_flag = 0  # because the winner_pose will be stored at the left side
                                # winner_flip_tags.append(1)
                                # loser_flip_tags.append(0)
                            elif result == 0:
                                loser_pose_centered[:, 0] = - loser_pose_centered[:, 0]
                                loser_crop = cv2.flip(loser_crop, 1)
                                flip_flag = 1 # [winner_pose, loser_pose]

                            cv2.imwrite(f"{export_path}/image_crops/pairs/{pair_id:05d}_v_{video_index:05d}_f_{frame_count:05d}_winner.jpg", winner_crop)
                            cv2.imwrite(f"{export_path}/image_crops/pairs/{pair_id:05d}_v_{video_index:05d}_f_{frame_count:05d}_loser.jpg", loser_crop)
                            crop_info.append([pair_id, video_index, frame_count, flip_flag, [winner_bbox, winner_frame_bbox], [loser_bbox, loser_frame_bbox]])
                            all_pose_pairs.append([winner_pose_centered, loser_pose_centered])
                            pair_id += 1
                    all_pose_pairs_per_video.append(all_pose_pairs) # N x 10 x 25 x 3
                    crop_info_per_video.append(crop_info)
            export_file_name = f"{export_path}/pose_crop_info/pose_pairs_centered_flipped_info.npy"
            with open(export_file_name, 'wb') as f:
                np.savez(f,
                         all_poses_centered=all_pose_pairs_per_video,
                         crop_info=crop_info_per_video
                         )

if __name__ == "__main__":
    crop_pose_image(database_path="/media/weihao/UNTITLED/tokyo_olymics/datasets/olympic_batch_2v3",
                    export_path="/media/weihao/UNTITLED/tokyo_olymics/datasets/olympic_batch_2v3/temp",
                    options=["pairs"],
                    window_offsets=[-10, 10])

