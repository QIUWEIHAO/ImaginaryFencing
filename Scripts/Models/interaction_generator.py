import torch
from torch import optim, nn
from Models.model_helpers import device, get_limb_conf, save_model



class Fight2(nn.Module):
    def __init__(self):
        super(Fight2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(52, 250),
            nn.Tanh(),
            nn.Linear(250, 500),
            nn.Tanh(),
            nn.Linear(500, 250),
            nn.Tanh(),
            nn.Linear(250, 50),
        )

    def forward(self, active_pose, dist):
        active_pose_keypoints = active_pose[:, :, 0:2].reshape(active_pose.shape[0], 50)
        fight_input = torch.cat([active_pose_keypoints, dist], dim=1)
        passive_pose = self.layers(fight_input).reshape(-1, 25, 2)
        return passive_pose


def train_fight2(training_data_pairs: [],
                 dists,
                 loss_func,
                 epoch: int = 60,
                 batch_size: int = 32,
                 lr: float = 0.03,
                 l2_lamda: float = 0.01,
                 save_models=False,
                 save_path="./"):
    model = Fight2().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2_lamda)
    active_poses_tensor = training_data_pairs[0]
    passive_poses_tensor = training_data_pairs[1]
    passive_poses_limb_conf = get_limb_conf(passive_poses_tensor)
    errors = []
    for e in range(epoch):
        batch_index = 0
        while (batch_index + batch_size) < active_poses_tensor.shape[0]:
            optimizer.zero_grad()
            passive_keypoints = model(active_poses_tensor[batch_index: batch_index + batch_size].float(),
                                      dists[batch_index: batch_index + batch_size].float())
            error = loss_func(passive_poses_tensor[batch_index: batch_index + batch_size],
                              passive_keypoints,
                              passive_poses_limb_conf[batch_index: batch_index + batch_size])
            error.backward()
            optimizer.step()

            batch_index += batch_size

        if (e + 1) % int((max(epoch, 10) / 10)) == 0 or e==0:
            print(f"Epoch: {(e + 1):05d}, Training Err:{error:.4f}")
            if save_models:
                save_model(model, save_path + f"{(e + 1):04d}.pth")

    return model
