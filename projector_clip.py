# Modified StyleGAN2 Projector with CLIP, addl. losses, kmeans, etc.
# by Peter Baylies, 2021 -- @pbaylies on Twitter

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import math
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
#import madgrad
import SM3
import clip

import dnnlib
import legacy

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

def score_images(G, model, text, latents, device, label_class = 0, batch_size = 8):
  scores = np.array([])
  all_images = np.array([])
  for i in range(math.ceil(latents.shape[0]/batch_size)):
    images = G.synthesis(torch.tensor(latents[i*batch_size:(i+1)*batch_size,:,:], dtype=torch.float32, device=device), noise_mode='const')
    with torch.no_grad():
        image_input = (torch.clamp(images, -1, 1) + 1) * 0.5
        image_input = F.interpolate(image_input, size=(256, 256), mode='area')
        image_input = image_input[:, :, 16:240, 16:240] # 256 -> 224, center crop
        image_input -= image_mean[None, :, None, None]
        image_input /= image_std[None, :, None, None]
        score = model(image_input, text)[0]
        scores = np.append(scores, score.cpu().numpy())
        all_images = np.append(all_images, images.cpu().numpy())

  scores = np.array(scores)
  #scores = np.hstack(scores).flatten()
  #scores = scores.reshape(-1, *scores.shape[2:]).squeeze()
  #scores = scores.reshape(-1).squeeze()
  #print(scores.shape)
  #print(scores)
  scores = 1 - scores / np.linalg.norm(scores)
  all_images = np.array(all_images)
  #all_images = all_images.reshape(-1, *all_images.shape[2:])
  return scores, all_images

def cluster_latents(samples, num_clusters, device):
    from kmeans_pytorch import kmeans
    # data
    #print(samples.shape)
    data_size = samples.shape[0]
    dims = samples.shape[2]
    x = torch.from_numpy(samples)

    # kmeans
    print(f'Performing kmeans clustering using {data_size} latents into {num_clusters} clusters...')
    cluster_ids_x, cluster_centers = kmeans(
        X=x, num_clusters=num_clusters, distance='euclidean', device=device
    )
    return cluster_centers, cluster_ids_x

def project(
    G,
    target_image: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    target_text,
    *,
    num_steps                  = 300,
    w_avg_samples              = 8192,
    initial_learning_rate      = 0.02,
    initial_latent             = None,
    initial_noise_factor       = 0.01,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.5,
    noise_ramp_length          = 0.5,
    latent_range               = 2.0,
    max_noise                  = 0.5,
    min_threshold              = 0.6,
    use_vgg                    = True,
    use_clip                   = True,
    use_pixel                  = True,
    use_penalty                = True,
    use_center                 = True,
    regularize_noise_weight    = 1e5,
    kmeans                     = True,
    kmeans_clusters            = 64,
    verbose                    = False,
    use_w_only                 = True,
    device: torch.device
):
    if target_image is not None:
        assert target_image.shape == (G.img_channels, G.img_resolution, G.img_resolution)
    else:
        use_vgg = False
        use_pixel = False

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.randn(w_avg_samples, G.z_dim)
    labels = None
    if (G.mapping.c_dim):
        labels = torch.from_numpy(0.5*np.random.randn(w_avg_samples, G.mapping.c_dim)).to(device)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), labels)  # [N, L, C]
    w_samples = w_samples.cpu().numpy().astype(np.float32)                 # [N, L, C]
    w_samples_1d = w_samples[:, :1, :].astype(np.float32)

    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, L, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    std_dev = np.std(w_samples)

    kmeans_latents = None
    if initial_latent is not None:
        w_avg = initial_latent
        if w_avg.shape[1] == 1 and not use_w_only:
            w_avg = np.tile(w_avg, (1, G.mapping.num_ws, 1))
    else:
        if kmeans and use_clip and target_text is not None:
            kmeans_latents, cluster_ids_x = cluster_latents(w_samples_1d, kmeans_clusters, device)
            cluster_centers = torch.tensor(kmeans_latents, dtype=torch.float32, device=device, requires_grad=True), cluster_ids_x

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    if use_vgg:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)

    # Load CLIP
    if use_clip:
        model, transform = clip.load("ViT-B/16", device=device)

    # Features for target image.
    if target_image is not None:
        target_images = target_image.unsqueeze(0).to(device).to(torch.float32)
        small_target = F.interpolate(target_images, size=(64, 64), mode='area')
        if use_center:
            center_target = F.interpolate(target_images, size=(448, 448), mode='area')[:, :, 112:336, 112:336]
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        target_images = target_images[:, :, 16:240, 16:240] # 256 -> 224, center crop

    if use_vgg:
        vgg_target_features = vgg16(target_images, resize_images=False, return_lpips=True)
        if use_center:
            vgg_target_center = vgg16(center_target, resize_images=False, return_lpips=True)

    if use_clip:
        if target_image is not None:
            with torch.no_grad():
                clip_target_features = model.encode_image(((target_images / 255.0) - image_mean[None, :, None, None]) / image_std[None, :, None, None]).float()
                if use_center:
                    clip_target_center = model.encode_image(((center_target / 255.0) - image_mean[None, :, None, None]) / image_std[None, :, None, None]).float()

    if kmeans and kmeans_latents is not None and use_clip and target_text is not None:
        scores, kmeans_images = score_images(G, model, target_text, kmeans_latents.repeat([1, G.mapping.num_ws, 1]), device=device)
        ind = np.argpartition(scores, 2)[:2]
        #w_avg = torch.median(kmeans_latents[ind],dim=0,keepdim=True)[0].repeat([1, G.mapping.num_ws, 1])

        filter_clusters = np.in1d(cluster_ids_x.cpu().numpy(), ind)
        filtered_latents = w_samples_1d[filter_clusters]
        kmeans_latents, cluster_ids_x = cluster_latents(filtered_latents, kmeans_clusters // 2, device)
        cluster_centers = torch.tensor(kmeans_latents, dtype=torch.float32, device=device, requires_grad=True), cluster_ids_x

        batch_size = 8
        if kmeans_latents.shape[0] < 8:
            batch_size = kmeans_latents.shape[0]
        scores, kmeans_images = score_images(G, model, target_text, kmeans_latents.repeat([1, G.mapping.num_ws, 1]), device=device, batch_size=batch_size)
        ind = np.argpartition(scores, 2)[:2]
        #w_avg = torch.median(kmeans_latents[ind],dim=0,keepdim=True)[0].repeat([1, G.mapping.num_ws, 1])

        filter_clusters = np.in1d(cluster_ids_x.cpu().numpy(), ind)
        final_latents = filtered_latents[filter_clusters]
        batch_size = 8
        if final_latents.shape[0] < 8:
            batch_size = final_latents.shape[0]
        scores, kmeans_images = score_images(G, model, target_text, np.tile(final_latents, (1, G.mapping.num_ws, 1)), device=device, batch_size=batch_size)
        ind = np.argpartition(scores, 1)[:1]
        final_candidates = torch.tensor(final_latents, dtype=torch.float32, device=device, requires_grad=True)
        w_avg = torch.median(final_candidates[ind],dim=0,keepdim=True)[0]
        if not use_w_only:
            w_avg = w_avg.repeat([1, G.mapping.num_ws, 1])
    else:
        if (use_w_only):
            w_avg = np.mean(w_avg, axis=1, keepdims=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_avg_tensor = w_opt.clone()
    with torch.no_grad():
        latent_range = torch.max(w_avg_tensor) + std_dev
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    #optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    #optimizer = madgrad.MADGRAD([w_opt] + list(noise_bufs.values()), lr=initial_learning_rate)
    optimizer = SM3.SM3([w_opt] + list(noise_bufs.values()), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = max_noise * w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        #print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        if use_w_only:
            ws = ws.repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(torch.clamp(ws,-latent_range,latent_range), noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. CLIP was built for 224x224 images.
        synth_images = (torch.clamp(synth_images, -1, 1) + 1) * (255/2)
        small_synth = F.interpolate(synth_images, size=(64, 64), mode='area')
        if use_center:
            center_synth = F.interpolate(synth_images, size=(448, 448), mode='area')[:, :, 112:336, 112:336]
        synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_images = synth_images[:, :, 16:240, 16:240] # 256 -> 224, center crop

        dist = 0

        if use_vgg:
            vgg_synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            vgg_dist =  (vgg_target_features - vgg_synth_features).square().sum()
            if use_center:
                vgg_synth_center = vgg16(center_synth, resize_images=False, return_lpips=True)
                vgg_dist += (vgg_target_center - vgg_synth_center).square().sum()
            vgg_dist *= 6
            dist += F.relu(vgg_dist*vgg_dist - min_threshold)

        if use_clip:
            clip_synth_image = ((synth_images / 255.0) - image_mean[None, :, None, None]) / image_std[None, :, None, None]
            clip_synth_features = model.encode_image(clip_synth_image).float()
            adj_center = 2.0

            if use_center:
                clip_cynth_center_image = ((center_synth / 255.0) - image_mean[None, :, None, None]) / image_std[None, :, None, None]
                adj_center = 1.0
                clip_synth_center = model.encode_image(clip_cynth_center_image).float()

            if target_image is not None:
                clip_dist =  (clip_target_features - clip_synth_features).square().sum()
                if use_center:
                    clip_dist += (clip_target_center - clip_synth_center).square().sum()
                dist += F.relu(0.5 + adj_center*clip_dist - min_threshold)

            if target_text is not None:
                clip_text = 1 - model(clip_synth_image, target_text)[0].sum() / 100
                if use_center:
                    clip_text += 1 - model(clip_cynth_center_image, target_text)[0].sum() / 100
                dist += 2*F.relu(adj_center*clip_text*clip_text - min_threshold / adj_center)

        if use_pixel:
            pixel_dist =  (target_images - synth_images).abs().sum() / 2000000.0
            if use_center:
                pixel_dist += (center_target - center_synth).abs().sum() / 2000000.0
            pixel_dist += (small_target - small_synth).square().sum() / 800000.0
            pixel_dist /= 4
            dist += F.relu(lr_ramp * pixel_dist - min_threshold)

        if use_penalty:
            #l1_penalty = (w_opt - w_avg_tensor).abs().sum() / 5000.0
            penalty_range = torch.sqrt(torch.arange(start=1,end=G.mapping.num_ws+1).float()).to(device)
            l1_penalty = ((w_opt - w_avg_tensor)*penalty_range[None, :, None]).abs().sum() / 20000.0
            l2_penalty = ((w_opt - w_avg_tensor)*penalty_range[None, :, None]).square().sum() / 10000.0

            dist += F.relu(lr_ramp * l1_penalty - min_threshold)
            dist += F.relu(lr_ramp * l2_penalty - min_threshold)

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        #print(vgg_dist, clip_dist, pixel_dist, l1_penalty, reg_loss * regularize_noise_weight)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
        #print(torch.max(w_opt))

        #if (torch.max(w_opt) > latent_range):
        #    with torch.no_grad():
        #        initial_learning_rate *= 0.9
        #        torch.add(w_opt, -w_avg_tensor, out=w_opt)
        #        torch.mul(w_opt, 0.8, out=w_opt)
        #        torch.add(w_opt, w_avg_tensor, out=w_opt)
        #        print(torch.max(w_opt))

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]
        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target-image', 'target_fname', help='Target image file to project to', required=False, metavar='FILE', default=None)
@click.option('--target-text',            help='Target text to project to', required=False, default=None)
@click.option('--initial-latent',         help='Initial latent', default=None)
@click.option('--lr',                     help='Learning rate', type=float, default=0.3, show_default=True)
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=300, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--use-vgg',                help='Use VGG16 in the loss', type=bool, default=True, show_default=True)
@click.option('--use-clip',               help='Use CLIP in the loss', type=bool, default=True, show_default=True)
@click.option('--use-pixel',              help='Use L1/L2 distance on pixels in the loss', type=bool, default=True, show_default=True)
@click.option('--use-penalty',            help='Use a penalty on latent values distance from the mean in the loss', type=bool, default=True, show_default=True)
@click.option('--use-center',             help='Optimize against an additional center image crop', type=bool, default=True, show_default=True)
@click.option('--min-threshold',          help='Minimum threshold for ReLU cutoff', required=False, default=0.6, show_default=True)
@click.option('--kmeans',                 help='Perform kmeans clustering for selecting initial latents', type=bool, default=True, show_default=True)
@click.option('--use-w-only',             help='Project into w space instead of w+ space', type=bool, default=False, show_default=True)
def run_projection(
    network_pkl: str,
    target_fname: str,
    target_text: str,
    initial_latent: str,
    outdir: str,
    save_video: bool,
    seed: int,
    lr: float,
    num_steps: int,
    use_vgg: bool,
    use_clip: bool,
    use_pixel: bool,
    use_penalty: bool,
    use_center: bool,
    min_threshold: float,
    kmeans: bool,
    use_w_only: bool,
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_image = None
    if target_fname:
        target_pil = PIL.Image.open(target_fname).convert('RGB').filter(PIL.ImageFilter.SHARPEN)

        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target_image = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)

    if target_text:
        target_text = torch.cat([clip.tokenize(target_text)]).to(device)

    if initial_latent is not None:
        initial_latent = np.load(initial_latent)
        initial_latent = initial_latent[initial_latent.files[0]]

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target_image=target_image,
        target_text=target_text,
        initial_latent=initial_latent,
        initial_learning_rate=lr,
        num_steps=num_steps,
        use_vgg=use_vgg,
        use_clip=use_clip,
        use_pixel=use_pixel,
        use_penalty=use_penalty,
        use_center=use_center,
        kmeans=kmeans,
        use_w_only=use_w_only,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            if use_w_only:
                synth_image = G.synthesis(projected_w.unsqueeze(0).repeat([1, G.mapping.num_ws, 1]), noise_mode='const')
            else:
                synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            if target_fname:
                video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
            else:
                video.append_data(synth_image)
        video.close()

    # Save final projected frame and W vector.
    if target_fname:
        target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    if use_w_only:
        synth_image = G.synthesis(projected_w.unsqueeze(0).repeat([1, G.mapping.num_ws, 1]), noise_mode='const')
    else:
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
