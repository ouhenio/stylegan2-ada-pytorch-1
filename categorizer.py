# Categorizer, by Peter Baylies (@pbaylies), 2021
# Unsupervised categorization of generated or real images using deep features, dimensionality reduction, and clustering
import click
from tqdm import tqdm
import math
import numpy as np
import torch
import pickle
import PIL.Image
import os.path
from torchvision.transforms import Compose
import torch.nn.functional as F
import clip
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# https://github.com/pratogab/batch-transforms
class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    def __init__(self):
        self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        if (not torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tensor.float().div_(self.max)


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor

from abc import ABC, abstractmethod

class BaseFeatureModel(ABC):
    # Get the model name at initialization time
    def __init__(self, name, device):
        self.name = name
        self.device = device
        super().__init__()

    # Return dimension of features returned by the model
    @property
    @abstractmethod
    def size(self):
        pass

    # Return expected image input size used by the model
    @property
    @abstractmethod
    def input_size(self):
        pass


    # Perform inference on an image, return features
    @abstractmethod
    def run(self, image):
        pass

class CLIPFeatureModel(BaseFeatureModel):
    def __init__(self, name, device):
        super().__init__(name, device)
        # Initialize the model
        self.model, _ = clip.load(self.name, device=self.device)
        self.transform = Compose([
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), inplace=True),
        ])

        # Feature embedding size and input size of currently released CLIP models computed below
        self.input_size = (224,224)
        if self.name == "RN50":
            self.size = 1024
        elif self.name == "RN50x4":
            self.size = 640
            self.input_size = (288,288)
        elif self.name == "RN50x16":
            self.size = 768
            self.input_size = (384,384)
        else:
            self.size = 512

    def size(self):
        return self.size

    def input_size(self):
        return self.input_size

    def run(self, image):
        image = self.transform(image).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(image)

    def encode_text(self, text):
        with torch.no_grad():
            text = clip.tokenize(text).to(self.device)
            return self.model.encode_text(text)

    def logits_per_image(self, image, text):
        image_features = self.run(image)
        with torch.no_grad():
            text_features = self.encode_text(text)
            logits_per_image, _ = self.model(image, text)
            return logits_per_image

    def softmax(self, image, text):
        logits_per_image = self.logits_per_image(image, text)
        return logits_per_image.softmax(dim=-1).cpu().numpy()

class TIMMFeatureModel(BaseFeatureModel):
    def __init__(self, name, device, out_indices = None):
        super().__init__(name, device)
        if out_indices is not None:
            self.model = timm.create_model(self.name, pretrained=True, features_only=True, num_classes=0, out_indices=out_indices).to(device)
        else:
            self.model = timm.create_model(self.name, pretrained=True, num_classes=0).to(device)
        self.model.eval()
        self.config = resolve_data_config({}, model=self.model)
        self.input_size = self.config['input_size'][1:]

        self.transform = create_transform(**self.config)
        self.transform = Compose([
            ToTensor(),
            Normalize(self.config['mean'], self.config['std'], inplace=True),
        ])
        out = self.run(torch.randn(self.config['input_size']).unsqueeze(0))
        self.size = out.shape[1]


    def size(self):
        return self.size

    def input_size(self):
        return self.input_size

    def run(self, image):
        with torch.no_grad():
            image = self.transform(image).to(self.device)
            out = self.model(image)
            if type(out) is list:
                flat = []
                for x in out:
                    flat.append(torch.nn.AvgPool2d(x.shape[2:])(x))
                return torch.cat(flat,dim=1).squeeze(dim=3).squeeze(dim=2)
            return out


device = torch.device('cuda')
loaded_models = {}

def get_files(path, ext = ''):
    from glob import glob
    return glob(path + '/*' + ext);

def generate_latents(G, num_samples):
    z_samples = np.random.randn(num_samples, G.z_dim)
    labels = None
    if (G.mapping.c_dim):
        labels = torch.from_numpy(0.2*np.random.randn(num_samples, G.mapping.c_dim)).to(device)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), labels)  # [N, L, C]
    w_samples = w_samples.cpu().numpy().astype(np.float32)                 # [N, L, C]
    return w_samples[:, :1, :].astype(np.float32).squeeze()

def load_images(files, size=(224,224)):
  images = []
  for file in files:
    images.append(PIL.Image.open(file).convert('RGB').resize(size, resample=PIL.Image.LANCZOS))
  return images

def convert_images(image_inputs):
  images = []
  for image in image_inputs:
    images.append(np.array(image).astype('float32'))
  return np.array(images).astype('float32')

def image_grid(images, rows, cols):
    assert len(images) <= rows*cols
    w, h = images[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(images):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def save_image_grids(all_images, max_grid_dim=8, outdir="", prefix=""):
    total_images = images_left = len(all_images)
    max_grid_num = max_grid_dim * max_grid_dim
    max_images = math.ceil(total_images / max_grid_num)
    last_image_saved = next_image_saved = grid_count = 0
    images_left = total_images - last_image_saved
    image_dim_size = max_grid_dim

    while images_left > 0:
        next_image_saved = last_image_saved + min(images_left, max_grid_num)
        if images_left < max_grid_num:
            image_dim_size = math.ceil(math.sqrt(images_left))
        image_grid(all_images[last_image_saved:next_image_saved], image_dim_size, image_dim_size).save(outdir + f"/{prefix}grid%06d.jpg" % grid_count)
        last_image_saved = next_image_saved
        images_left = total_images - last_image_saved
        grid_count += 1

def run_pca(components, features, outdir=""):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(components, len(features)))
    pca.fit(features)
    if outdir:
        pickle.dump( pca, open( outdir + "/pca_model.pkl", "wb" ) )
    return pca.transform(features)

def run_ica(components, features, outdir="", max_iter=400):
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=min(components, len(features)), max_iter=max_iter)
    ica.fit(features)
    if outdir:
        pickle.dump( ica, open( outdir + "/ica_model.pkl", "wb" ) )
    return ica.transform(features)

def fit_gmm(components, features, covariance_type='tied', outdir="", max_iter=200):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=components, covariance_type=covariance_type, verbose=2, max_iter=max_iter)
    gmm.fit(features)
    if outdir:
        pickle.dump( gmm, open( outdir + "/gmm_model.pkl", "wb" ) )
    return gmm.predict(features)

def compute_center_clusters(features, labels, num_categories, num_features):
    avg = np.zeros((num_categories, num_features))
    count = np.zeros(num_categories)
    for f, l in zip(features, labels):
        avg[l] += f
        count[l] += 1
    cnt = 0
    for c in np.nditer(count):
        avg[cnt] = avg[cnt] / (c + 0.00000001)
        cnt += 1
    return avg

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=False)
@click.option('--dataset',                help='Dataset path', required=False)
@click.option('--verbose',                help='Display more information', type=bool, default=True, show_default=True)
@click.option('--num-samples',            help='Number of images to cluster', type=int, default=8192, show_default=True)
@click.option('--num-categories',         help='Number of total clusters', type=int, default=64, show_default=True)
@click.option('--num-subcategories',      help='Size of subclusters', type=int, default=0, show_default=True)
@click.option('--filter-by-label',        help='Filter dataset by a given cluster label', type=int, default=-1, show_default=True)
@click.option('--batch-size',             help='Batch size', type=int, default=16, show_default=True)
@click.option('--use-latents',            help='Use latents (if available) as features', type=bool, default=True, show_default=True)
@click.option('--use-clip-models',        help='Use CLIP models for producing features', default='ViT-B/16', show_default=True)
@click.option('--use-timm-models',        help='Use timm models for producing features', default='', show_default=True)
@click.option('--reduce-with-pca',        help='Reduce features with n dimensions of PCA (or 0 for off)', type=int, default=256, show_default=True)
@click.option('--reduce-with-ica',        help='Reduce features with n dimensions of ICA (or 0 for off)', type=int, default=256, show_default=True)
@click.option('--use-pca-bottleneck',     help='Reduce features again with n dimensions of PCA (or 0 for off)', type=int, default=128, show_default=True)
@click.option('--gmm-covariance-type',    help='Covariance type of GMM to use (options are full, tied, diag, spherical)', default='tied', show_default=True)
@click.option('--resume-dir',             help='Where to load/reuse compatible intermediate data', required=False, metavar='DIR')
@click.option('--outdir',                 help='Where to save the output images and intermediate data', required=True, metavar='DIR')
def run_categorization(
    network_pkl: str,
    dataset: str,
    verbose: bool,
    num_samples: int,
    num_categories: int,
    num_subcategories: int,
    filter_by_label: int,
    batch_size: int,
    use_latents: bool,
    use_clip_models: str,
    use_timm_models: str,
    reduce_with_pca: int,
    reduce_with_ica: int,
    use_pca_bottleneck: int,
    gmm_covariance_type: str,
    resume_dir: str,
    outdir: str,
):
    G = None
    w = None
    files = None
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if (network_pkl): # if we're generating images from StyleGAN2
        import dnnlib
        import legacy
        if verbose:
            print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as fp:
            G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
        if (resume_dir and os.path.isfile(resume_dir + "/latents.npy")):
            if verbose:
                print('Loading saved latents...')
            w = np.load(resume_dir + "/latents.npy")
        else:
            if verbose:
                print('Generating %d latents...' % num_samples)
            w = generate_latents(G, num_samples)
        np.save(outdir + "/latents.npy", w)
    else:
        if (dataset):
            if verbose:
                print('Loading dataset file list...')
            files = get_files(dataset)
            if num_samples <= 0:
                num_samples = len(files)
            else:
                files = files[0:num_samples]

    model_classes = []
    features = {}
    model_size = {}

    if (resume_dir and os.path.isfile(resume_dir + "/all_features.npy")):
        if (resume_dir and os.path.isfile(resume_dir + "/more_features.npy")):
            if verbose:
                print("Skipping all features...")
        else:
            if verbose:
                print('Loading all features...')
            all_features = np.load(resume_dir + "/all_features.npy")
    else:
        if (use_clip_models):
            for model_name in use_clip_models.split(','):
                if verbose:
                    print('Initializing CLIP model %s' % model_name)
                model_classes.append(CLIPFeatureModel(model_name, device))

        if (use_timm_models):
            for model_info in use_timm_models.split(','):
                model_features = None
                if ":" in model_info:
                    model_name, model_features = model_info.split(':')
                else:
                    model_name = model_info
                if model_features is not None:
                    model_features = model_features.split('|')
                    model_features = [int(i) for i in model_features]
                if verbose:
                    print('Initializing TIMM model %s' % model_name)
                model_classes.append(TIMMFeatureModel(model_name, device, model_features))


        if model_classes:
            if verbose:
                print('Computing features...')
            if num_samples < batch_size:
                batch_size = num_samples
            else:
                if num_samples % batch_size != 0:
                    batch_size = math.gcd(batch_size, num_categories)
            for i in tqdm(range(num_samples//batch_size)):
                images = None
                image_input = None

                if G:
                    with torch.no_grad():
                        images = G.synthesis(torch.tensor(np.tile(np.expand_dims(w[i*batch_size:(i+1)*batch_size,:],axis=1),[1,G.mapping.num_ws,1]), dtype=torch.float32, device=device), noise_mode='const')
                        image_batch = (torch.clamp(images, -1, 1) + 1) * 127.5

                for m in model_classes:
                    if (not m.name in features):
                        features[m.name] = []
                    if (not m.name in model_size):
                        model_size[m.name] = m.size

                    if dataset:
                        images = load_images(files[i*batch_size:(i+1)*batch_size], size=m.input_size)
                        image_input = np.transpose(convert_images(images), (0, 3, 1, 2))
                    else:
                        with torch.no_grad():
                            image_input = F.interpolate(image_batch, size=m.input_size, mode='area').cpu().numpy()
                    features[m.name].append(m.run(image_input).cpu().numpy())

        all_logits = []
        if w is not None:
            all_logits.append(w)

        for m in model_classes:
            logits = np.array(features[m.name])
            logits = logits.reshape(-1, *logits.shape[2:]).squeeze()
            all_logits.append(logits)
            logits = None

        all_features = np.concatenate(all_logits, axis=1)
        del all_logits

        np.save(outdir + "/all_features.npy", all_features)

    if (resume_dir and os.path.isfile(resume_dir + "/more_features.npy")):
        if verbose:
            print('Loading reduced features...')
        more_features = np.load(resume_dir + "/more_features.npy")
    else:
        if reduce_with_pca or reduce_with_ica:
            reduced_features = []

            if reduce_with_pca:
                if verbose:
                    print('Running PCA with %d features...' % reduce_with_pca)
                reduced_features.append(run_pca(reduce_with_pca, all_features, outdir=outdir))

            if reduce_with_ica:
                if verbose:
                    print('Running ICA with %d features...' % reduce_with_ica)
                reduced_features.append(run_ica(reduce_with_ica, all_features, outdir=outdir))

            more_features = np.concatenate(reduced_features, axis=1)
            del reduced_features
        else:
            more_features = all_features

        if use_pca_bottleneck:
            if verbose:
                print('Running PCA bottleneck with %d features...' % use_pca_bottleneck)
            more_features = run_pca(use_pca_bottleneck, more_features, outdir=outdir)

    np.save(outdir + "/more_features.npy", more_features)

    if (resume_dir and os.path.isfile(resume_dir + "/labels.npy")):
        if verbose:
            print('Loading labels...')
        labels = np.load(resume_dir + "/labels.npy")
    else:
        if verbose:
            print('Computing %d labels with %s GMM' % (num_categories, gmm_covariance_type))
        labels = fit_gmm(num_categories, more_features, covariance_type=gmm_covariance_type, outdir="")

    np.save(outdir + "/labels.npy", labels)

    prefix = ''
    if (filter_by_label > -1):
        if num_subcategories > 0:
            num_categories = num_subcategories
        print('Filtering by label #%d with %d clusters' % (filter_by_label, num_categories))
        prefix = "l%d_" % filter_by_label
        more_features = more_features[labels == filter_by_label]
        if G:
            w = w[labels == filter_by_label]
        else:
            files = np.array(files)[labels == filter_by_label]
        labels = fit_gmm(num_categories, more_features, covariance_type=gmm_covariance_type)

    if (resume_dir and os.path.isfile(resume_dir + f"/{prefix}avg.npy")):
        if verbose:
            print('Loading average cluster centers...')
        avg = np.load(resume_dir + f"/{prefix}avg.npy")
    else:
        if verbose:
            print('Computing %d average cluster centers...' % num_categories)
        if G:
            avg = compute_center_clusters(w, labels, num_categories, G.z_dim)
        else:
            avg = compute_center_clusters(more_features, labels, num_categories, more_features.shape[1])
    np.save(outdir + f"/{prefix}avg.npy", avg)

    if G:
        if verbose:
            print('Generating images for %d cluster centers...' % num_categories)
        all_images = []
        if num_categories < batch_size:
            batch_size = num_categories
        else:
            if num_categories % batch_size != 0:
                batch_size = math.gcd(batch_size, num_categories)
        for i in range(avg.shape[0]//batch_size):
            images = G.synthesis(torch.tensor(np.tile(np.expand_dims(avg[i*batch_size:(i+1)*batch_size,:], axis=1),[1,G.mapping.num_ws,1]), dtype=torch.float32, device=device), noise_mode='const')
            img = (images.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy() * 127.5 + 128).astype(np.uint8)
            for j in range(img.shape[0]):
                all_images.append(PIL.Image.fromarray(img[j]))
    else:
        if verbose:
            print('Finding images for %d cluster centers...' % num_categories)
        from sklearn.metrics import pairwise_distances_argmin_min
        closest_files = []
        for count, avg in enumerate(avg):
            close, _ = pairwise_distances_argmin_min(np.expand_dims(avg, axis=0), more_features[labels == count])
            close = close[0]
            closest_files.append(np.array(files)[labels == count][close])
        all_images = load_images(closest_files, size=(512,512))

    if verbose:
        print('Saving image grid(s)')
    save_image_grids(all_images, outdir=outdir, prefix=prefix)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_categorization() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
