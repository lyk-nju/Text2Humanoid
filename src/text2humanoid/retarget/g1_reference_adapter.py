from __future__ import annotations

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.retarget.fk_features import (
    build_local_body_features,
    build_world_features,
    load_dataset_joint_names,
    remap_nmr_dof_to_dataset,
)


class G1ReferenceAdapter:
    def __init__(
        self,
        xml_path: str | None = None,
        tracking_config_path: str | None = None,
    ) -> None:
        self.xml_path = xml_path
        self._tracking_config_path = tracking_config_path
        self.joint_names = load_dataset_joint_names(tracking_config_path)

    def from_nmr_result(
        self,
        chunk_id: str,
        start_time: float,
        fps: int,
        result: dict,
    ) -> G1ReferenceChunk:
        root_pos = np.asarray(result["root_trans"], dtype=np.float32)
        root_rot_wxyz = np.asarray(result["root_rot_quat"], dtype=np.float32)
        root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]].astype(np.float32)

        dof_pos_dataset = remap_nmr_dof_to_dataset(np.asarray(result["dof"], dtype=np.float32))
        body_pos_w, body_rot_w, body_names = build_world_features(
            root_pos=root_pos,
            root_rot_xyzw=root_rot_xyzw,
            dof_pos_dataset=dof_pos_dataset,
            xml_path=self.xml_path,
        )
        local_body_pos, local_body_rot = build_local_body_features(
            root_pos=root_pos,
            root_rot_xyzw=root_rot_xyzw,
            body_pos_w=body_pos_w,
            body_rot_w=body_rot_w,
        )

        return G1ReferenceChunk(
            chunk_id=chunk_id,
            start_time=start_time,
            fps=fps,
            root_pos=root_pos,
            root_rot=root_rot_xyzw,
            dof_pos=dof_pos_dataset,
            local_body_pos=local_body_pos,
            local_body_rot=local_body_rot,
            body_names=body_names,
            joint_names=list(self.joint_names),
            metadata={
                "source": "MakeTrackingEasy",
                "root_quat_order": "xyzw",
                "body_rot_order": "xyzw",
            },
        )
