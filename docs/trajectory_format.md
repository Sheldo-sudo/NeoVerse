# Trajectory Format Specification

This document describes how to define camera trajectories in NeoVerse.

## Overview

NeoVerse supports several ways to specify camera trajectories:

1. **Predefined Trajectory (CLI)** — Choose from 13 built-in trajectory types via `--trajectory`
2. **Keyframe Operations (JSON)** — Compose operations at keyframe timestamps
3. **Sparse Matrices (JSON)** — Provide camera-to-world 4x4 matrices at sparse frame indices
4. **NPZ File Reference (JSON)** — Load matrices from an external `.npz` file

Methods 2–4 use a JSON file passed via `--trajectory_file`.

## Method 1: Predefined Trajectory

Use the `--trajectory` flag to select a built-in trajectory type:

```bash
python inference.py --trajectory pan_left --input_path input.mp4 --output_path outputs/pan_left.mp4
```

### Available Types

| Type | Description | Parameters |
|------|-------------|------------|
| `pan_left` | Rotate camera left (Y-axis) | `angle` |
| `pan_right` | Rotate camera right (Y-axis) | `angle` |
| `tilt_up` | Rotate camera up (X-axis) | `angle` |
| `tilt_down` | Rotate camera down (X-axis) | `angle` |
| `move_left` | Translate camera left (-X) | `distance` |
| `move_right` | Translate camera right (+X) | `distance` |
| `push_in` | Translate camera forward (+Z) | `distance` |
| `pull_out` | Translate camera backward (-Z) | `distance` |
| `boom_up` | Translate camera up (-Y) | `distance` |
| `boom_down` | Translate camera down (+Y) | `distance` |
| `orbit_left` | Arc counter-clockwise in XZ plane | `angle`, `orbit_radius` |
| `orbit_right` | Arc clockwise in XZ plane | `angle`, `orbit_radius` |
| `static` | No movement (identity) | — |

### Parameter Overrides

Override default parameters with CLI flags:

```bash
# Pan left by 45 degrees instead of the default 30
python inference.py --trajectory pan_left --angle 45 --input_path input.mp4

# Push in with larger distance
python inference.py --trajectory push_in --distance 0.3 --input_path input.mp4

# Orbit with custom radius and angle
python inference.py --trajectory orbit_left --angle 45 --orbit_radius 0.8 --input_path input.mp4

# Set trajectory mode and zoom
python inference.py --trajectory pan_left --traj_mode global --zoom_ratio 1.2 --input_path input.mp4
```

### Default Parameter Values

| Parameter | Default |
|-----------|---------|
| `angle` | 15 (degrees) |
| `distance` | 0.1 |
| `orbit_radius` | 1.0 |

## Method 2: Keyframe Operations (JSON)

Define trajectories as a sequence of keyframes, each with one or more camera operations.

### Schema

```json
{
  "name": "My Trajectory",
  "mode": "relative",
  "num_frames": 81,
  "keyframes": [
    {"0": [{"static": {}}]},
    {"40": [{"pan_left": {"angle": 15}}]},
    {"80": [{"pan_left": {"angle": 30}}]}
  ]
}
```

Each keyframe is a single-key dict: `{timestamp: [operations]}`.

Each operation is a single-key dict: `{type: params}`.

### Operation Types

All 13 predefined types are available as operations. Each takes a parameters dict:

| Operation | Parameters | Description |
|-----------|-----------|-------------|
| `pan_left` | `{"angle": 15}` | Rotate left around Y-axis |
| `pan_right` | `{"angle": 15}` | Rotate right around Y-axis |
| `tilt_up` | `{"angle": 15}` | Rotate up around X-axis |
| `tilt_down` | `{"angle": 15}` | Rotate down around X-axis |
| `move_left` | `{"distance": 0.1}` | Translate along -X |
| `move_right` | `{"distance": 0.1}` | Translate along +X |
| `push_in` | `{"distance": 0.1}` | Translate along +Z |
| `pull_out` | `{"distance": 0.1}` | Translate along -Z |
| `boom_up` | `{"distance": 0.1}` | Translate along -Y |
| `boom_down` | `{"distance": 0.1}` | Translate along +Y |
| `orbit_left` | `{"angle": 15, "orbit_radius": 1.0}` | Arc left in XZ plane |
| `orbit_right` | `{"angle": 15, "orbit_radius": 1.0}` | Arc right in XZ plane |
| `static` | `{}` | Identity (no motion) |

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `angle` | float | 15 | Rotation angle in degrees |
| `distance` | float | 0.1 | Translation distance (scene-relative units) |
| `orbit_radius` | float | 1.0 | Radius of orbital arc |

### Keyframe Rules

1. **Timestamps**: String keys in JSON, converted to float internally. Range `[0, num_frames - 1]`.
2. **Ascending order**: Keyframe timestamps must be strictly ordered.
3. **First and last**: First keyframe must have timestamp `"0"`, last must have `"num_frames - 1"`.
4. **Minimum count**: At least 2 keyframes required.
5. **Interpolation**: Rotations interpolated via SLERP, translations via linear interpolation.
6. **Accumulation**: Operations at each keyframe are composed onto the running accumulated pose.

### Compound Motion

Multiple operations can be specified in a single keyframe to create compound motions. They are composed via matrix multiplication in list order:

```json
{
  "name": "Orbit with Tilt",
  "mode": "relative",
  "num_frames": 81,
  "keyframes": [
    {"0": [{"static": {}}]},
    {"80": [
      {"orbit_left": {"angle": 15, "orbit_radius": 1.0}},
      {"tilt_up": {"angle": 10}}
    ]}
  ]
}
```

## Method 3: Sparse Matrices (JSON)

Provide camera-to-world 4x4 matrices at sparse frame indices. Intermediate frames are interpolated automatically.

### Schema

```json
{
  "name": "sparse_matrices",
  "mode": "relative",
  "num_frames": 81,
  "trajectory": {
    "frame_indices": [0, 40, 80],
    "frame_matrices": [
      [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ],
      [
        [0.9397, 0.0, -0.3420, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.3420, 0.0, 0.9397, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ],
      [
        [0.9397, 0.0, 0.3420, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-0.3420, 0.0, 0.9397, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    ]
  }
}
```

### Fields

- `frame_indices`: List of integer frame indices (ascending, at least 2 entries, non-negative).
- `frame_matrices`: List of 4x4 camera-to-world matrices, one per frame index. Length must match `frame_indices`.

### Interpolation

Sparse keyframes are interpolated to a dense trajectory of `num_frames` frames:

- **Rotations**: SLERP (Spherical Linear Interpolation)
- **Translations**: Linear interpolation

### Matrix Format

Each matrix is a 4x4 camera-to-world transformation in OpenCV convention (X-right, Y-down, Z-forward):

```
[R | t]   R = 3x3 rotation matrix (orthonormal, det=1)
[0 | 1]   t = 3x1 translation vector (camera position)
```

See [coordinate_system.md](coordinate_system.md) for axis definitions and matrix details.

## Method 4: NPZ File Reference (JSON)

Reference an external `.npz` file containing pre-computed camera matrices.

### Schema

```json
{
  "name": "custom_traj",
  "mode": "relative",
  "num_frames": 81,
  "trajectory": {
    "file": "/path/to/custom.npz"
  }
}
```

### NPZ File Contents

The `.npz` file must contain two arrays:

| Array | Shape | Description |
|-------|-------|-------------|
| `frame_indices` | `(N,)` | Integer frame indices |
| `frame_matrices` | `(N, 4, 4)` | Camera-to-world transformation matrices |

### Creating an NPZ File

```python
import numpy as np

frame_indices = np.array([0, 40, 80])
frame_matrices = np.zeros((3, 4, 4), dtype=np.float32)

# Frame 0: identity
frame_matrices[0] = np.eye(4)

# Frame 40: some pose
frame_matrices[1] = np.eye(4)
frame_matrices[1, 0, 3] = 0.1  # translate right

# Frame 80: another pose
frame_matrices[2] = np.eye(4)
frame_matrices[2, 0, 3] = 0.2  # translate right more

np.savez_compressed("/path/to/cameras.npz",
                    frame_indices=frame_indices,
                    frame_matrices=frame_matrices)
```

## Common JSON Fields

These fields are shared across all JSON-based methods (2, 3, and 4):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | filename stem | Human-readable trajectory name |
| `description` | string | — | Optional description |
| `mode` | string | `"relative"` | `"relative"` or `"global"` (see below) |
| `num_frames` | integer | 81 | Number of output frames |
| `zoom_ratio` | float | 1.0 | Zoom factor (1.0 = no zoom) |
| `use_first_frame` | boolean | false | When the target trajectory starts from the same viewpoint as the first input frame, replace the rendered first frame with the original input frame for better quality |

### Mode

- **`"relative"`** (default): Trajectory transformations are applied relative to the original camera motion: `T_target = T_original × T_trajectory`. Recommended for most use cases.
- **`"global"`**: Trajectory is used as-is in world coordinates: `T_target = T_trajectory`. Useful for precise positioning.

## Validation

Validate a trajectory file without running inference:

```bash
python inference.py --trajectory_file your_trajectory.json --validate_only
```

Output includes format type, entry count, and mode. Common validation errors:

- **Invalid JSON syntax** — missing commas, brackets, or quotes
- **Missing fields** — `keyframes` or `trajectory` must be present
- **Bad keyframe structure** — each keyframe must be a single-key dict `{timestamp: [operations]}`
- **Invalid timestamps** — must be in `[0, num_frames-1]`, ascending, first=0, last=num_frames-1
- **Unknown operation type** — must be one of the 13 valid types
- **Invalid matrix dimensions** — each matrix must be exactly 4x4 with numeric values
- **Mismatched lengths** — `frame_indices` and `frame_matrices` must have the same length

## Examples

### Example Files

The repository includes ready-to-use trajectory files:

**`examples/trajectories/`**:
- `orbit_left_pull_out.json` — Compound keyframe trajectory (orbit + pull out)
- `sparse_matrices.json` — Inline sparse matrix trajectory
- `custom.json` + `custom.npz` — NPZ file reference trajectory


---

For more information, see:
- [coordinate_system.md](coordinate_system.md) — Coordinate system and matrix reference
- [examples/trajectories/](../examples/trajectories/) — Example trajectory files
