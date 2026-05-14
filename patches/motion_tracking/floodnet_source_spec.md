# FloodNet Source Plugin Spec

`motion_tracking` should gain a minimal source plugin that accepts online
reference frames from `Text2Humanoid`.

## Required Frame Payload

Each pushed frame should contain:

- `root_pos`: `(3,)`, world xyz
- `root_quat`: `(4,)`, xyzw
- `dof_pos`: `(29,)`, dataset joint order

## Required Clip Payload

The clip-level payload should contain:

- `fps`
- `root_pos`
- `root_rot`
- `dof_pos`
- `local_body_pos`
- `local_body_rot`
- `body_names`
- `joint_names`

## Non-goals

- no policy retraining
- no network structure changes
- no dataset changes inside `motion_tracking`
