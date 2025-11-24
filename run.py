import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from manopth.manolayer import ManoLayer

# ===== wandb: import =====
import wandb

# ===== wandb: init =====
wandb.init(
    # project="dex-retarget-mlp",  # 换成你自己的项目名
    # name="mano_pose2joint_mlp",  # 这次实验的 run 名
    project="DexVAE",
    dir="output",
    config={
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "num_epochs": 200,
        "val_ratio": 0.1,
        "hidden_dims": [512, 512, 512],
    },
    name="mano_pose2joint_mlp",
)
config = wandb.config

data_x = np.load(
    "/home/ghr/panwei/pw-workspace/dex-retargeting/data/original_loader/data_x.npz"
)["data_x"]  # [N, mano_param_dim]
data_label = np.load(
    "/home/ghr/panwei/pw-workspace/dex-retargeting/data/original_loader/data_label.npz"
)["data_label"]  # [N, num_robot_joints]

print("data_x shape:", data_x.shape)
print("data_label shape:", data_label.shape)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_x_torch = torch.from_numpy(data_x).float().to(device)
data_x_torch = data_x_torch.squeeze(1)
data_label_torch = torch.from_numpy(data_label).float().to(device)

mano_layer = ManoLayer(
    mano_root="/home/ghr/panwei/pw-workspace/dex_latent/dex-ycb-toolkit/manopth/mano/models",
    use_pca=True,
    ncomps=15,
    flat_hand_mean=False,
).to(device)

random_shape = torch.rand(data_x_torch.shape[0], 10, device=device)

with torch.no_grad():
    _, _, _, th_rot_map_train, th_rot_map_6d_train = mano_layer(
        data_x_torch, random_shape
    )

th_rot_map_6d_train = th_rot_map_6d_train[:,6:] # global rotation
x = th_rot_map_6d_train.reshape(th_rot_map_6d_train.shape[0], -1)
y = data_label_torch[:, 6:] #abandon 6 float joint
# import ipdb; ipdb.set_trace()

print("MLP input x shape:", x.shape)  # [N, D_in]
print("MLP target y shape:", y.shape)  # [N, D_out]

# ========= 3. 划分 train / val =========
N = x.shape[0]
val_ratio = config.val_ratio
val_size = int(N * val_ratio)
train_size = N - val_size

perm = torch.randperm(N, device=device)
train_idx = perm[:train_size]
val_idx = perm[train_size:]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)


from network.network import Pose2JointMLP

model = Pose2JointMLP(
    in_dim=x.shape[1],
    out_dim=y.shape[1],
    hidden_dims=config.hidden_dims,
).to(device)
print(model)


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, weight_decay=config.weight_decay
)

# ===== wandb: watch model =====
wandb.watch(model, log="all", log_freq=50)

num_epochs = config.num_epochs
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    train_loss /= train_size

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= val_size

    # ===== wandb: per-epoch log =====
    wandb.log(
        {
            "epoch": epoch,
            "loss/train_mse": train_loss,
            "loss/val_mse": val_loss,
        }
    )

    if epoch % 5 == 0 or epoch == 1:
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
        )
    
    # save every 50 epochs:
    if epoch % 50 == 0:
        save_path = f"/home/ghr/panwei/pw-workspace/dex-retargeting/data/mlp_retarget_{epoch}.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "in_dim": x.shape[1],
                "out_dim": y.shape[1],
            },
            save_path,
        )

        print(f"Saved MLP model to: {save_path}")


# ===== wandb: 结束 run（可选） =====
wandb.finish()
