import torch
import numpy as np

BODY_STRUCT = np.array(
    [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11], [11, 22], [22, 23],
     [11, 24], [8, 12], [12, 13], [13, 14], [14, 19], [19, 20], [14, 21], [0, 15], [15, 17], [0, 16], [16, 18]])
device = "cuda:0"

def get_limb_conf(poses):
    keypoint_conf = poses[:, :, 2]
    limb_conf = torch.zeros([poses.shape[0], BODY_STRUCT.shape[0]]).to(device)

    # for each limb, the confidence of the limb equals 0 if either joint is 0, or the average if none is zero
    for j in range(BODY_STRUCT.shape[0]):
        for i in range(poses.shape[0]):
            if (keypoint_conf[i, BODY_STRUCT[j, 0]] == 0) | (keypoint_conf[i, BODY_STRUCT[j, 1]] == 0):
                limb_conf[i, j] = 0
            else:
                limb_conf[i, j] = (keypoint_conf[i, BODY_STRUCT[j, 0]] + keypoint_conf[i, BODY_STRUCT[j, 1]]) / 2
    return limb_conf


def prepare_nd_to_torch(ndarray, dim=0):
    # transfer to device
    tensor = torch.from_numpy(ndarray).float().to(device)
    # get mean and std
    std, mean = torch.std_mean(tensor[:, :, 0:2], dim=dim)
    # normalize
    tensor[:, :, 0:2] = (tensor[:, :, 0:2] - mean) / std
    return std, mean, tensor


def restore_torch_to_nd(tensor, std, mean):
    tensor_copy = tensor.clone().detach()
    tensor_copy[:, :, 0:2] = tensor_copy[:, :, 0:2] * std + mean
    return tensor_copy.cpu().numpy()

def save_model(model, filename):
    torch.save(model, filename)


def load_model(filename):
    model = torch.load(filename).to(device)
    return model