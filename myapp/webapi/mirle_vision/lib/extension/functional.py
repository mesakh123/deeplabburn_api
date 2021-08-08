from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms.functional import normalize


def beta_smooth_l1_loss(input: Tensor, target: Tensor, beta: float) -> Tensor:
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    loss = loss.sum() / (input.numel() + 1e-8)
    return loss


def normalize_means_stds(input: Tensor, means: List[float], stds: List[float]) -> Tensor:
    assert input.ndim in [3, 4]

    num_channels = input.shape[-3]
    assert len(means) == len(stds) == num_channels

    if input.ndim == 3:
        return normalize(input, means, stds)
    else:
        return torch.stack([normalize(it, means, stds) for it in input], dim=0)


def denormalize_means_stds(input: Tensor, means: List[float], stds: List[float]) -> Tensor:
    assert input.ndim in [3, 4]

    num_channels = input.shape[-3]
    assert len(means) == len(stds) == num_channels

    if input.ndim == 3:
        output = normalize(
            normalize(
                input,
                mean=(0, 0, 0), std=[1 / v for v in stds]),
            mean=[-v for v in means], std=(1, 1, 1)
        )
        return output
    else:
        return torch.stack([
            normalize(
                normalize(
                    it,
                    mean=(0, 0, 0), std=[1 / v for v in stds]),
                mean=[-v for v in means], std=(1, 1, 1)
            )
            for it in input
        ], dim=0)


def crop_and_resize(image_batch: Tensor,
                    norm_crop_center_x_batch: Tensor, norm_crop_center_y_batch: Tensor,
                    norm_crop_width_batch: Tensor, norm_crop_height_batch: Tensor,
                    resized_width: int, resized_height: int) -> Tensor:
    assert image_batch.ndim == 4
    assert norm_crop_center_x_batch.ndim == 1
    assert norm_crop_center_y_batch.ndim == 1
    assert norm_crop_width_batch.ndim == 1
    assert norm_crop_height_batch.ndim == 1
    assert ((norm_crop_center_x_batch >= 0) & (norm_crop_center_x_batch <= 1)).all().item()
    assert ((norm_crop_center_y_batch >= 0) & (norm_crop_center_y_batch <= 1)).all().item()
    assert ((norm_crop_width_batch >= 0) & (norm_crop_width_batch <= 1)).all().item()
    assert ((norm_crop_height_batch >= 0) & (norm_crop_height_batch <= 1)).all().item()

    batch_size, _, image_height, image_width = image_batch.shape

    resized_crop_batch = []
    for b in range(batch_size):
        image = image_batch[b]
        norm_crop_center_x = norm_crop_center_x_batch[b]
        norm_crop_center_y = norm_crop_center_y_batch[b]
        norm_crop_width = norm_crop_width_batch[b]
        norm_crop_height = norm_crop_height_batch[b]

        crop_width = int(image_width * norm_crop_width)
        crop_height = int(image_height * norm_crop_height)

        norm_crop_left = norm_crop_center_x - norm_crop_width / 2
        norm_crop_top = norm_crop_center_y - norm_crop_height / 2
        x_samples = torch.linspace(start=0, end=1, steps=crop_width).to(norm_crop_width) * norm_crop_width + norm_crop_left
        y_samples = torch.linspace(start=0, end=1, steps=crop_height).to(norm_crop_height) * norm_crop_height + norm_crop_top

        grid = torch.meshgrid(x_samples, y_samples)
        grid = torch.stack(grid, dim=-1)
        grid = grid.transpose(0, 1)
        grid = grid * 2 - 1

        crop_batch = F.grid_sample(input=image.unsqueeze(dim=0),
                                   grid=grid.unsqueeze(dim=0),
                                   mode='bilinear',
                                   align_corners=True)
        resized_crop = F.interpolate(input=crop_batch,
                                     size=(resized_height, resized_width),
                                     mode='bilinear',
                                     align_corners=True).squeeze(dim=0)
        resized_crop_batch.append(resized_crop)
    resized_crop_batch = torch.stack(resized_crop_batch, dim=0)
    return resized_crop_batch


if __name__ == '__main__':
    def main():
        def test_crop_and_resize():
            image_batch = torch.tensor([[1, 1, 2, 2],
                                        [1, 1, 2, 2],
                                        [3, 3, 4, 4],
                                        [3, 3, 4, 4]],
                                       dtype=torch.float, requires_grad=True).unsqueeze(dim=0).unsqueeze(dim=0)
            norm_crop_center_x_batch = torch.tensor([0.625], dtype=torch.float, requires_grad=True)
            norm_crop_center_y_batch = torch.tensor([0.75], dtype=torch.float, requires_grad=True)
            norm_crop_width_batch = torch.tensor([0.75], dtype=torch.float, requires_grad=True)
            norm_crop_height_batch = torch.tensor([0.5], dtype=torch.float, requires_grad=True)

            resized_crop_batch = crop_and_resize(image_batch,
                                                 norm_crop_center_x_batch, norm_crop_center_y_batch,
                                                 norm_crop_width_batch, norm_crop_height_batch,
                                                 resized_width=4, resized_height=4)

            print('image_batch:\n', image_batch)
            print('resized_crop_batch:\n', resized_crop_batch)

            image_batch.retain_grad()
            norm_crop_center_x_batch.retain_grad()
            norm_crop_center_y_batch.retain_grad()
            norm_crop_width_batch.retain_grad()
            norm_crop_height_batch.retain_grad()
            resized_crop_batch.retain_grad()

            resized_crop_batch.sum().backward()

            print('image_batch.grad:\n', image_batch.grad)
            print('norm_crop_center_x_batch.grad:\n', norm_crop_center_x_batch.grad)
            print('norm_crop_center_y_batch.grad:\n', norm_crop_center_y_batch.grad)
            print('norm_crop_width_batch.grad:\n', norm_crop_width_batch.grad)
            print('norm_crop_height_batch.grad:\n', norm_crop_height_batch.grad)
            print('resized_crop_batch.grad:\n', resized_crop_batch.grad)

        test_crop_and_resize()

    main()
