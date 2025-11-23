import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import sapien
from hand_viewer import HandDatasetSAPIENViewer
from pytransform3d import rotations
from tqdm import trange

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

from dex_latent.utils import dcm2rv

import torch

class RobotHandDatasetSAPIENViewer(HandDatasetSAPIENViewer):
    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
        headless=False,
        use_ray_tracing=False,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type

        # Load optimizer and filter
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # Build robot
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )
            urdf_name = urdf_path.name
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)
            self.retarget2sapien.append(retarget2sapien)

    def load_object_hand(self, data: Dict):
        super().load_object_hand(data)
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        for _ in range(len(self.robots)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self._load_ycb_object(ycb_id, ycb_mesh_file)

    def render_dexycb_data(self, data: Dict, fps=5, y_offset=0.8, device: str = "cuda", sample_from_model: bool = False, model_path: str = None):
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))
            self.camera.set_local_pose(local_pose)

        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        num_frame = hand_pose.shape[0]
        num_copy = len(self.robots) + 1
        num_ycb_objects = len(data["ycb_ids"])
        pose_offsets = []
        
        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        if not sample_from_model:
            for i in range(0, num_frame):
                init_hand_pose_frame = hand_pose[i]
                vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
                if vertex is not None:
                    start_frame = i
                    break

        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]
            robot_names = "_".join(robot_names)
            video_path = (
                Path(__file__).parent.resolve() / f"data/{robot_names}_video.mp4"
            )
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (self.camera.get_width(), self.camera.get_height()),
            )

        # Warm start
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(
            hand_pose_start[0, 0:3]
        )
        if not sample_from_model:
            vertex, joint = self._compute_hand_geometry(hand_pose_start)
            for robot, retargeting, retarget2sapien in zip(
                self.robots, self.retargetings, self.retarget2sapien
            ):
                retargeting.warm_start(
                    joint[0, :],
                    wrist_quat,
                    hand_type=self.hand_type,
                    is_mano_convention=True,
                )

        # Loop rendering
        step_per_frame = int(60 / fps)
        images_list = []
        # from manopth.manolayer import ManoLayer
        # from mano_layer import MANOLayer
        
        # mano_layer = MANOLayer(
        #     mano_root="dex-ycb-toolkit/manopth/mano/models",
        #     use_pca=True,
        #     ncomps=15,
        #     flat_hand_mean=False,
        # )
        # from manopth.manolayer import ManoLayer
        # self.mano_layer = ManoLayer(
        #     mano_root="/home/ghr/panwei/pw-workspace/dex_latent/dex-ycb-toolkit/manopth/mano/models",
        #     use_pca=True,
        #     ncomps=15,
        #     flat_hand_mean=False,
        # )
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            # hand_pose_frame = hand_pose[i]
            if sample_from_model:
                # sampler_hand_pose = self.sampler_hand_pose(
                #     device=device, model_path=model_path
                # )
                # random_shape = torch.rand(1, 10)
                # vertex, joint = self.mano_layer._mano_layer.verts_from_full_pose(
                #     sampler_hand_pose, random_shape,
                # )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                z = torch.randn(1, 12, device=device)
                from dex_latent.utils import rotation_6d_to_matrix

                with torch.no_grad():
                    use_pipeline = False
                    if not use_pipeline:
                        if not hasattr(self, 'model'):
                            self.load_model(device=device, model_path=model_path)
                        x_recon = self.model.decode(z)  # (1,90)
                        R = (
                            rotation_6d_to_matrix(x_recon.view(1, 15, 6))[0].cpu().numpy()
                        )  # (15,3,3)
                        random_shape = torch.rand(1, 10)

                        sample_rot_map = R.reshape(1, 15 * 9)
                        th_sample_rot_map = torch.from_numpy(sample_rot_map)

                        fixed_root_uniform_rot_map = torch.tensor(
                            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
                        )
                        # fixed_root_uniform_rot_map = torch.tensor(
                            # [[0.8660254, -0.5, 0.0, 0.5, 0.8660254, 0.0, 0.0, 0.0, 1.0]]
                        # )

                        th_rot_map = torch.cat(
                            [fixed_root_uniform_rot_map, th_sample_rot_map], 1
                        )
                        # trans = torch.tensor([[0.37241126, -0.037277655,  0.0755586]])
                        # trans = torch.zeros(1, 3)
                        vertex, joint = self.mano_layer._mano_layer.verts_from_full_pose(
                            th_rot_map, random_shape
                        )
                        print(f"vertex: {vertex.mean()}" , f"joint: {joint.mean()}")
                        vertex = vertex / 1000
                        joint = joint / 1000

                        vertex = vertex.squeeze(0)
                        joint = joint.squeeze(0).cpu().numpy()

                    else:
                        sampler_hand_pose = self.sampler_hand_pose(
                            device=device, model_path=model_path
                        )
                        vertex, joint = self._compute_hand_geometry(sampler_hand_pose)

            else:
                sampler_hand_pose = hand_pose[i]
                vertex, joint = self._compute_hand_geometry(sampler_hand_pose)
                # numpy: (0, 788,3) --> (788,3)
                # vertex = vertex[0]
            # Update poses for YCB objects
            # for k in range(num_ycb_objects):
            #     pos_quat = object_pose_frame[k]

            #     # Quaternion convention: xyzw -> wxyz
            #     pose = self.camera_pose * sapien.Pose(
            #         pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]])
            #     )
            #     self.objects[k].set_pose(pose)
            #     for copy_ind in range(num_copy):
            #         self.objects[k + copy_ind * num_ycb_objects].set_pose(
            #             pose_offsets[copy_ind] * pose
            #         )

            # Update pose for human hand
            # self._update_hand(vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(
                self.robots, self.retargetings, self.retarget2sapien
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)

            self.scene.update_render()
            if self.headless:
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                # writer.write(rgb[..., ::-1])
                images_list.append(rgb)

            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            import imageio.v2 as iio

            # video_path = Path(__file__).parent.resolve() / "data/human_hand_video.mp4"

            # iio.mimwrite(
            #     "/home/ghr/panwei/pw-workspace/dex-retargeting/README.mp4",
            #     [images[i] for images in images_list],
            # )
            iio.mimwrite(
                "/home/ghr/panwei/pw-workspace/dex-retargeting/README_shadow.mp4",
                images_list,  # 直接写整个列表
                fps=fps,  # 顺便把 fps 写清楚
            )
            print(
                f"saved video to /home/ghr/panwei/pw-workspace/dex-retargeting/README_shadow.mp4"
            )

    def load_model(self, device, model_path: str = None):
        from dex_latent.network import DexBetaVAE

        model = DexBetaVAE(
            input_dim=90, latent_dim=12, beta=1.0
        ).to(device)

        state = torch.load(
            model_path,
            map_location=device,
        )
        model.load_state_dict(state)
        model.eval()
        self.model = model

    def sampler_hand_pose(self, device, model_path: str = None):
        from dex_latent.utils import rotation_6d_to_matrix
        z = torch.randn(1, 12, device=device)

        # images_list = []

        if not hasattr(self, 'model'):
            self.load_model(device=device, model_path=model_path)

        with torch.no_grad():
            x_recon = self.model.decode(z)  # (1,90)
            # if not args.load_from_data:
            print(f"loading data from data!!!")

            R = (
                rotation_6d_to_matrix(x_recon.view(1, 15, 6))[0].cpu().numpy()
            )  # (15,3,3)
            # sample_rot_map = R.reshape(1, 15 * 9)
            # th_sample_rot_map = torch.from_numpy(sample_rot_map)
            fixed_root_uniform_rot_map_matrix = np.array(
                [[0.5, 0.8660254, 0.0], [0.8660254, -0.5, 0.0], [0.0, 0.0, 1.0]]
            )
            root_R = fixed_root_uniform_rot_map_matrix[None, :, :]
            hand_pose_frame = dcm2rv(torch.from_numpy(
                np.concatenate([  
                    root_R,
                    R, 
                ], axis=0).astype(np.float32))
            )
            hand_pose_frame = hand_pose_frame.reshape(1, 48)
            hand_pose_frame = torch.cat([hand_pose_frame, torch.tensor([[0.20241126, -0.17277655,  0.7755586]])], 1)
            return hand_pose_frame.cpu().numpy()
