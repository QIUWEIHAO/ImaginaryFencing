import torch
from torch import nn
import torch.optim as optim
from loss_function import compare_two_pose
from model_helpers import save_model, get_limb_conf

device = "cuda:0"


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(50, 25),
            nn.Tanh(),
            nn.Linear(25, 10),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 25),
            nn.Tanh(),
            nn.Linear(25, 50)
        )

    def encode(self, pose):
        x = pose[:, :, 0:2].reshape(pose.shape[0], 50)
        return self.encoder(x)

    def decode(self, latent):
        keypoints = self.decoder(latent).reshape(latent.shape[0], 25, 2)
        return keypoints

    def forward(self, pose):
        x = self.encode(pose)
        keypoints = self.decode(x)
        # print(keypoints.shape)
        output = torch.cat([keypoints, input[:, :, 2:3]], dim=2).to(device)
        return output

def test_ae(model, testing_data, loss_func) -> float:
    error = loss_func(testing_data, model(testing_data), get_limb_conf(testing_data))
    return error


def train_ae(training_data, testing_data, loss_func, epoch: int = 60, batch_size: int = 32,
             lr: float = 0.03, save_models=False, save_path='./') -> nn.Module:
    model = AE().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    limb_conf_training = get_limb_conf(training_data)
    for e in range(epoch):
        batch_index = 0
        while (batch_index + batch_size) < training_data.shape[0]:
            optimizer.zero_grad()
            batch_data = training_data[batch_index: batch_size + batch_index]
            batch_limb_conf = limb_conf_training[batch_index: batch_size + batch_index]
            error = loss_func(batch_data, model(batch_data), batch_limb_conf)
            error.backward()
            optimizer.step()

            batch_index += batch_size

        # print(f"epoch: {e:04d}, batch_index: {batch_index:04d}")
        if (e + 1) % int((max(epoch, 10) / 10)) == 0:
            e1 = test_ae(model, testing_data, loss_func)
            e2 = test_ae(model, training_data, loss_func)
            print(f"Epoch: {(e + 1):05d}, Training Err:{e2:.4f},  Testing Err:{e1:.4f}")
            if save_models:
                save_model(model, save_path + f"{(e + 1):04d}.pth")
    return model