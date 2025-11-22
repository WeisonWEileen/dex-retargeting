import torch
# https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_%E2%86%94_Euler_axis/angle
# https://github.com/kashif/ceres-solver/blob/087462a90dd1c23ac443501f3314d0fcedaea5f7/include/ceres/rotation.h#L178
# S. Sarabandi and F. Thomas. A Survey on the Computation of Quaternions from Rotation Matrices. J MECH ROBOT, 2019.
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def dcm2rv(dcm):
    """Converts direction cosine matrices to rotation vectors.

    Args:
      dcm: A tensor of shape [B, 3, 3] containing the direction cosine matrices.

    Returns:
      A tensor of shape [B, 3] containing the rotation vectors.
    """
    X = torch.stack(
        (
            dcm[:, 2, 1] - dcm[:, 1, 2],
            dcm[:, 0, 2] - dcm[:, 2, 0],
            dcm[:, 1, 0] - dcm[:, 0, 1],
        ),
        dim=1,
    )
    s = torch.norm(X, p=2, dim=1) / 2
    c = (dcm[:, 0, 0] + dcm[:, 1, 1] + dcm[:, 2, 2] - 1) / 2
    c = torch.clamp(c, -1, 1)
    angle = torch.atan2(s, c)
    Y = torch.stack((dcm[:, 0, 0], dcm[:, 1, 1], dcm[:, 2, 2]), dim=1)
    Y = torch.sqrt((Y - c.unsqueeze(1)) / (1 - c.unsqueeze(1)))
    rv = torch.zeros((dcm.size(0), 3), device=dcm.device)
    i1 = s > 1e-3
    i2 = (s <= 1e-3) & (c > 0)
    i3 = (s <= 1e-3) & (c < 0)
    rv[i1] = angle[i1].unsqueeze(1) * X[i1] / (2 * s[i1].unsqueeze(1))
    rv[i2] = X[i2] / 2
    rv[i3] = angle[i3].unsqueeze(1) * torch.sign(X[i3]) * Y[i3]
    return rv
