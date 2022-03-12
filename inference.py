import os
import argparse
import torch
import csv
import configs.data_config as data_config
import configs.full_cv_config as tr_test_config
from utils import augmentations as aug
from utils.data_loader import CDNet2014Loader
from utils import losses
from models.unet import unet_vgg16
from utils.eval_utils import logVideos

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BSUV-Net-2.0 pyTorch')
    parser.add_argument('--network', metavar='Network', dest='network', type=str,
                        default='R2AttU_temporal_network_attention_flux_without_median_bg_segmentation_ch',
                        help='Which network to use. unetvgg16, unet_attention, unet_attention_flux,'
                             'unet_no_empty_attention, unet_no_empty_attention_flux, unet_temporal_network_attention,'
                             'unet_temporal_network_attention_flux, unet_temporal_network, sparse_unet_flux'
                             'R2AttU_temporal_network_attention_flux'
                             'R2AttU_temporal_network_attention_flux_without_median_bg_segmentation_ch')

    # Input images
    parser.add_argument('--inp_size', metavar='Input Size', dest='inp_size', type=int, default=224,
                        help='Size of the inputs. If equals 0, use the original sized images. '
                             'Assumes square sized input')
    parser.add_argument('--empty_bg', metavar='Empty Background Frame', dest='empty_bg', type=str, default='no',
                        help='Which empty background to use? no, manual or automatic')
    parser.add_argument('--recent_bg', metavar='Recent Background Frame', dest='recent_bg', type=int, default=0,
                        help='Use recent background frame as an input as well. 0 or 1')
    parser.add_argument('--seg_ch', metavar='Segmentation', dest='seg_ch', type=int, default=0,
                        help='Whether to use the FPM channel input or not. 0 or 1')
    parser.add_argument('--flux_ch', metavar='Flux tensor', dest='flux_ch', type=int, default=1,
                        help='Whether to use the flux tensor input or not. 0 or 1')
    parser.add_argument('--current_fr', metavar='Current Frame', dest='current_fr', type=int, default=1,
                        help='Whether to use the current frame or not. 0 or 1')

    # Optimization
    parser.add_argument('--lr', metavar='Learning Rate', dest='lr', type=float, default=1e-4,
                        help='learning rate of the optimization')
    parser.add_argument('--weight_decay', metavar='weight_decay', dest='weight_decay', type=float, default=1e-2,
                        help='weight decay of the optimization')
    parser.add_argument('--num_epochs', metavar='Number of epochs', dest='num_epochs', type=int, default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', metavar='Minibatch size', dest='batch_size', type=int, default=1,
                        help='Number of samples per minibatch')
    parser.add_argument('--loss', metavar='Loss function to be used', dest='loss', type=str, default='jaccard',
                        help='Loss function to be used ce for cross-entropy or jaccard for Jaccard index')
    parser.add_argument('--opt', metavar='Optimizer to be used', dest='opt', type=str, default='adam',
                        help='sgd, rmsprop or adam')

    # Data augmentations
    parser.add_argument('--aug_noise', metavar='Data Augmentation for noise', dest='aug_noise', type=int, default=1,
                        help='Whether to use Data Augmentation for noise. 0 or 1')
    parser.add_argument('--aug_rsc', metavar='Data Augmentation for randomly-shifted crop', dest='aug_rsc', type=int,
                        default=1,
                        help='Whether to use randomly-shifted crop. 0 or 1')
    parser.add_argument('--aug_ptz', metavar='Data Augmentation for PTZ camera crop', dest='aug_ptz', type=int,
                        default=1,
                        help='Whether to use PTZ camera crop 0 or 1')
    parser.add_argument('--aug_id', metavar='Data Augmentation for Illumination Difference', dest='aug_id', type=int,
                        default=1,
                        help='Whether to use Data Augmentation for Illumination Difference. 0 or 1')
    parser.add_argument('--aug_ioa', metavar='Data Augmentation for Intermittent Object Addition', dest='aug_ioa',
                        type=float, default=0.1,
                        help='Probability of applying Intermittent Object Addition')

    # Cross-validation
    # You can select more than one train-test split, specify the id's of them
    parser.add_argument('--set_number', metavar='Which training-test split to use from config file', dest='set_number',
                        type=int, default=[1], help='Training and test videos will be selected based on the set number')

    # Model name
    parser.add_argument('--model_name', metavar='Name of the model for log keeping', dest='model_name',
                        type=str, default='BSUV-Net 2.0',
                        help='Name of the model to be used in output csv and checkpoints')

    args = parser.parse_args()
    network = args.network
    empty_bg = args.empty_bg
    current_fr = args.current_fr
    use_flux_tensor = args.flux_ch
    recent_bg = True if args.recent_bg == 1 else False
    seg_ch = True if args.seg_ch == 1 else False
    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    loss = args.loss
    opt = args.opt
    inp_size = args.inp_size
    if inp_size == 0:
        inp_size = None
    else:
        inp_size = (inp_size, inp_size)

    use_temporal_network = False
    temporal_length = 50
    temporal_kernel_size = 8

    aug_noise = args.aug_noise
    aug_rsc = args.aug_rsc
    aug_ptz = args.aug_ptz
    aug_id = args.aug_id
    aug_ioa = args.aug_ioa

    set_number = args.set_number
    cuda = True

    # naming for log keeping
    fname = args.model_name + "_network_" + network

    save_dir = data_config.save_dir
    mdl_dir = os.path.join(save_dir, fname)

    if network == "unet_attention_flux" or network == "unet_no_empty_attention_flux":
        use_flux_tensor = True

    if network == "unet_temporal_network" or "unet_temporal_network_attention":
        use_temporal_network = True

    # Evaluation on test videos
    model = torch.load(f"{mdl_dir}/model_best.mdl").cuda()
    csv_path = "./" + fname + "_log.csv"

    # Locations of each video in the CSV file
    csv_header2loc = data_config.csv_header2loc

    category_row = [0] * csv_header2loc['len']
    category_row[0] = 'category'

    scene_row = [0] * csv_header2loc['len']
    scene_row[0] = 'scene'

    metric_row = [0] * csv_header2loc['len']
    metric_row[0] = 'metric'

    for cat, vids in tr_test_config.datasets_tr[0].items():
        for vid in vids:
            category_row[csv_header2loc[vid]] = cat
            category_row[csv_header2loc[vid] + 1] = cat
            category_row[csv_header2loc[vid] + 2] = cat

            scene_row[csv_header2loc[vid]] = vid
            scene_row[csv_header2loc[vid] + 1] = vid
            scene_row[csv_header2loc[vid] + 2] = vid

            metric_row[csv_header2loc[vid]] = 'FNR'
            metric_row[csv_header2loc[vid] + 1] = 'Prec'
            metric_row[csv_header2loc[vid] + 2] = 'F-score'

    with open(csv_path, mode='w', newline="") as log_file:
        employee_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(category_row)
        employee_writer.writerow(scene_row)
        employee_writer.writerow(metric_row)

    # Inference selected train-test splits in the config file
    for idx in set_number:
        dataset_test = tr_test_config.datasets_test[idx]
        logVideos(
            dataset_test,
            model,
            fname,
            csv_path,
            current_fr=current_fr,
            empty_bg=empty_bg,
            use_flux_tensor=use_flux_tensor,
            use_temporal_network=use_temporal_network,
            temporal_length=temporal_length,
            recent_bg=recent_bg,
            segmentation_ch=seg_ch,
            save_vid=False,
            set_number=set_number,
            debug=False
        )

    print(f"Saved results to {csv_path}")