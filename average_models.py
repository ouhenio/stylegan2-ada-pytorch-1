# Stealing this one back, thanks aydao! -- pbaylies, 2021
#
# https://github.com/aydao/stylegan2-surgery/blob/master/avg_local.py
#   ~~~ aydao ~~~~ 2020 ~~~
#
#   Based on pbaylies' swa.py script
#   except that this computes the average instead of the moving average
#   and does so locally in this script, rather than by modifying network.py
#
import os
import glob
import pickle
import argparse

import torch

def add_networks(dst_net, src_net):
    params1 = src_net.named_parameters()
    params2 = dst_net.named_parameters()
    dict_params2 = dst_net.state_dict()
    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data + dict_params2[name1].data)
    dst_net.load_state_dict(dict_params2)
    return dst_net

def apply_denominator(dst_net, denominator):
    denominator_inv = 1.0 / denominator
    params = dst_net.named_parameters()
    dict_params = dst_net.state_dict()
    for name, param in params:
        dict_params[name].data.copy_(dict_params[name].data / denominator)
    dst_net.load_state_dict(dict_params)
    return dst_net

def main(args):

    filepath = args.output_model
    files = glob.glob(os.path.join(args.results_dir,args.filespec))
    files.sort()
    network_count = len(files)
    print('Discovered %s networks' % str(network_count))

    avg_kwargs, avg_G, avg_D, avg_Gs, aug = None, None, None, None, None
    for pkl_file in files:
        models = pickle.load(open(pkl_file, 'rb'))
        kwargs = models['training_set_kwargs']
        G = models['G']
        D = models['D']
        Gs = models['G_ema']
        aug = models['augment_pipe']
        if avg_G == None:
            avg_kwargs, avg_G, avg_D, avg_Gs, avg_aug = kwargs, G, D, Gs, aug
        else:
            avg_G = add_networks(avg_G, G)
            avg_D = add_networks(avg_D, D)
            avg_Gs = add_networks(avg_Gs, Gs)

    apply_denominator(avg_G, network_count)
    apply_denominator(avg_D, network_count)
    apply_denominator(avg_Gs, network_count)

    models = {'training_set_kwargs': avg_kwargs, 'G': avg_G, 'D': avg_D, 'G_ema': avg_Gs, 'augment_pipe': avg_aug}

    print('Final model parameters set to weight average.')
    with open(filepath, 'wb') as f:
        pickle.dump(models, f)
    print('Final averaged weights saved to file.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform weight averaging', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results_dir', help='Directory with network checkpoints for weight averaging')
    parser.add_argument('--filespec', default='*.pkl', help='The files to average')
    parser.add_argument('--output_model', default='network-avg.pkl', help='The averaged model to output')

    args = parser.parse_args()

    main(args)
