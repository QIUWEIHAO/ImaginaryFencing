import torch
from model_helpers import BODY_STRUCT

def compare_two_pose(pose_i, pose_t, limb_conf):
    # print(f"pose_i: {pose_i.shape} , pose_t: {pose_t.shape}")
    # limb_conf = get_limb_conf(pose_i)
    limb_start = BODY_STRUCT[:, 0]
    limb_end = BODY_STRUCT[:, 1]

    a1 = (pose_i[:, limb_start, 0:2] - pose_t[:, limb_start, 0:2]) \
        .reshape(pose_i.shape[0], BODY_STRUCT.shape[0], 2)
    a2 = (pose_i[:, limb_end, 0:2] - pose_t[:, limb_end, 0:2]) \
        .reshape(pose_i.shape[0], BODY_STRUCT.shape[0], 2)

    error = torch.sum(torch.sum(a1 * a1 + a2 * a2, axis=2) * limb_conf) / BODY_STRUCT.shape[0] / pose_i.shape[0]
    # print(f"limb_conf:{limb_conf}")
    return error

