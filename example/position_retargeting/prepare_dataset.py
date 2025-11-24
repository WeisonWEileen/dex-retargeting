from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tyro

from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer

def save_dataset(
    robots: Optional[Tuple[RobotName]],
    data_root: Path,
    fps: int,
    device: str = "cuda",
    sample_from_model: bool = False,
    model_path: str = None,
):
    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    if robots is None:
        viewer = HandDatasetSAPIENViewer(headless=True)
    else:
        viewer = RobotHandDatasetSAPIENViewer(
            list(robots), HandType.right, headless=True
        )

    # Data ID, feel free to change it to visualize different trajectory
    data_x = []
    data_label = []
    for data_id in range(len(dataset)):
        # data_id = 4

        sampled_data = dataset[data_id]
        for key, value in sampled_data.items():
            if "pose" not in key:
                print(f"{key}: {value}")
        viewer.load_object_hand(sampled_data)
        # viewer.render_dexycb_data(sampled_data, fps)

        data_x_i, data_label_i = viewer.get_retargeted_data(
            sampled_data,
            fps,
            device=device,
            sample_from_model=sample_from_model,
            model_path=model_path,
        )

        data_x.extend(data_x_i)
        data_label.extend(data_label_i)

    np.savez(f"data_x.npz", data_x=data_x)
    np.savez(f"data_label.npz", data_label=data_label)
    return None


def main(
    dexycb_dir: str,
    robots: Optional[List[RobotName]] = None,
    fps: int = 10,
    device: str = "cuda",
    sample_from_model: bool = False,
    model_path: str = None,
):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    The human trajectory is visualized as provided, while the robot trajectory is generated from position retargeting

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to render, if None, render human hand trajectory only
        fps: frequency to render hand-object trajectory

    """
    data_root = Path(dexycb_dir).absolute()
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")

    # args = tyro.cli(main)
    save_dataset(
        robots,
        data_root,
        fps,
        device=device,
        sample_from_model=sample_from_model,
        model_path=model_path,
    )


if __name__ == "__main__":
    tyro.cli(main)
