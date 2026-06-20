import PIL.Image
import numpy as np
import torch
import spandrel
import tqdm
from spandrel import ImageModelDescriptor
from PIL.Image import Image

from lib.upscaling.util import image_to_tensor, tensor_to_image


def upscale_image_spandrel(image: Image, model_path, upscale_factor=4, progress=None):
    # Load the model
    loader = spandrel.ModelLoader()
    model = loader.load_from_file(model_path)

    # make sure it's an image to image model
    assert isinstance(model, ImageModelDescriptor)

    # send it to the GPU and put it in inference mode
    model.cuda().eval()

    # convert the image to a tensor
    image_tensor = image_to_tensor(np.array(image)).to('cuda')

    # use the model
    with torch.no_grad():
        out = tiled_scale(image_tensor, model, tile_x=64, tile_y=64, overlap=8, upscale_amount=upscale_factor,
                          out_channels=3,
                          progress=progress)

    out = torch.clamp(out, min=0, max=1.0)
    return PIL.Image.fromarray(tensor_to_image(out))


def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, out_channels=3,
                output_device="cpu", progress=None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount),
                          round(samples.shape[3] * upscale_amount)), device=output_device)

    sample_steps = 0
    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                sample_steps += 1

    pbar = None
    if progress is not None:
        pbar = progress.tqdm(range(0, sample_steps))

    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros(
            (s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)),
            device=output_device)
        out_div = torch.zeros(
            (s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)),
            device=output_device)
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                x = max(0, min(s.shape[-1] - overlap, x))
                y = max(0, min(s.shape[-2] - overlap, y))
                s_in = s[:, :, y:y + tile_y, x:x + tile_x]

                ps = function(s_in).to(output_device)
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    mask[:, :, t:1 + t, :] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, mask.shape[2] - 1 - t: mask.shape[2] - t, :] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, :, t:1 + t] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, :, mask.shape[3] - 1 - t: mask.shape[3] - t] *= ((1.0 / feather) * (t + 1))
                out[:, :, round(y * upscale_amount):round((y + tile_y) * upscale_amount),
                round(x * upscale_amount):round((x + tile_x) * upscale_amount)] += ps * mask
                out_div[:, :, round(y * upscale_amount):round((y + tile_y) * upscale_amount),
                round(x * upscale_amount):round((x + tile_x) * upscale_amount)] += mask
                if pbar is not None:
                    pbar.update(1)
        output[b:b + 1] = out / out_div
    return output
