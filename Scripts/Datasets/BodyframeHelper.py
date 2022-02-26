import numpy as np
import cv2
import torch

BODY_STRUCT = np.array(
    [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11], [11, 22], [22, 23],
     [11, 24], [8, 12], [12, 13], [13, 14], [14, 19], [19, 20], [14, 21], [0, 15], [15, 17], [0, 16], [16, 18]])

# Fiding bounding box for given keypoints (screen space) of a skeleton
def find_bbox(keypoints: [], screen_dim: [] = [1280, 720]) -> []:
    keypoints_copy = keypoints
    to_delete = []
    cnt = 0
    for kp in keypoints_copy:
        pos = (int(kp[0]), int(kp[1]))
        if (pos == (0, 0)) or (pos == (screen_dim[0], 0)):
            to_delete.append(cnt)
        cnt += 1
    keypoints_copy = np.delete(keypoints_copy, to_delete, 0)
    return [min(keypoints_copy[:, 0]), max(keypoints_copy[:, 0]), min(keypoints_copy[:, 1]), max(keypoints_copy[:, 1])]

# find the largest two skeletons from all detected skeletons as the two players
def get_player_skeletons(bodyframes: []) -> []:
    all_bbox = []
    all_conf = []
    for bf in bodyframes:
        keypoints = np.array(bf['pose_keypoints_2d']).reshape(25, -1)
        all_bbox.append(find_bbox(keypoints))
        all_conf.append(sum(keypoints[:, 2]))

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

        kp1 = np.array(bodyframes[top2_bf[0]]['pose_keypoints_2d']).reshape(25, 3)
        kp2 = np.array(bodyframes[top2_bf[1]]['pose_keypoints_2d']).reshape(25, 3)

        center_0 = sum(np.array(find_bbox(kp1))[0:2])
        center_1 = sum(np.array(find_bbox(kp2))[0:2])
        if center_0 > center_1:
            left_bf = top2_bf[1]
            right_bf = top2_bf[0]

        # Save pose_keypoints
        pose_keypoints_left = np.array(bodyframes[left_bf]['pose_keypoints_2d']).reshape(25, 3)
        pose_keypoints_right = np.array(bodyframes[right_bf]['pose_keypoints_2d']).reshape(25, 3)
        return [pose_keypoints_left, pose_keypoints_right]
    elif len(top2_bf) == 1:
        print(f"only 1 skeletons were found")
        return [np.array(bodyframes[top2_bf[0]]['pose_keypoints_2d']).reshape(25, 3)]
    else:
        print(f"No skeletons were found")
        return []

# draw skeleton keypoints(screen resolution range) on the image
def draw_keypoints(frame, keypoints, color=(255, 255, 255), thickness=3, draw_confidence=True):
    width = frame.shape[1]
    for b, e in BODY_STRUCT:
        pos1 = (int(keypoints[b][0]), int(keypoints[b][1]))
        pos2 = (int(keypoints[e][0]), int(keypoints[e][1]))
        if (pos1 != (0, 0)) & (pos2 != (0, 0)) & (pos1 != (width, 0)) & (pos2 != (width, 0)):
            frame = cv2.line(frame, pos1, pos2, color, thickness)

    if draw_confidence:
        confidence = np.sum(keypoints[:, 2])
        bbox = find_bbox(keypoints)
        #         print("confidence: {:.2f}".format(confidence))
        frame = cv2.putText(frame, "conf: {:.2f}".format(confidence), (int(bbox[0]), int(bbox[2]) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=6)
    return frame

# draw multiple skeletons on the image, map the color based on confidence
def visualize_poses(canvas, poses, line_color, map_color: bool = True, line_thickness = 3):
    width = canvas.shape[1]
    height = canvas.shape[0]
    for i in range(poses.shape[0]):
        pose = poses[i].copy()

        pose[:, 0] = pose[:, 0] * width
        pose[:, 1] = pose[:, 1] * height

        # if pose have the confidence channel
        if pose.shape[1] == 3 and map_color:
            mapped_color = tuple([((sum(pose[:, 2]) - 8) / 17) * x for x in line_color])
        else:
            mapped_color = tuple([x for x in line_color])
        # print(mapped_color)
        draw_keypoints(canvas, pose, color=mapped_color, draw_confidence=False, thickness=line_thickness)
    return canvas

# given multiple skeletons (0-1 range), find their centers of bounding boxes, the centered pose and the bounding box
def find_pose_centers(poses, screen_dims: [] = [1, 1]):
    centers = []
    bboxes = []
    poses_centered = np.zeros_like(poses)
    poses_screen_space = np.zeros_like(poses)
    poses_screen_space[:, :, 0] = poses[:, :, 0] * screen_dims[0]
    poses_screen_space[:, :, 1] = poses[:, :, 1] * screen_dims[1]
    if poses.shape[2] == 3:
        poses_screen_space[:, :, 2] = poses[:, :, 2]
    for i in range(poses.shape[0]):
        bbox = find_bbox(poses_screen_space[i, :, :])
        bboxes.append(bbox)
        center = [(bbox[0] + bbox[1]) / 2.0 / screen_dims[0], (bbox[2] + bbox[3]) / 2.0 / screen_dims[1]]
        #         print(bbox[0])
        centers.append(center)
        poses_centered[i, :, 0] = poses[i, :, 0] - center[0]
        poses_centered[i, :, 1] = poses[i, :, 1] - center[1]
    centers = np.array(centers)
    bboxes = np.array(bboxes)
    return centers, poses_centered, bboxes
