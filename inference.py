import os
import argparse
import torch
import csv
import configs.data_config as data_config
import configs.full_cv_config as tr_test_config
from models.unet import unet_vgg16
from models.sparse_unet import FgNet
from models.unet_attention import AttU_Net, R2AttU_Net
from models.convlstm_network import SEnDec_cnn_lstm
from utils import augmentations as aug
from utils.data_loader import CDNet2014Loader
from utils import losses
from models.unet import unet_vgg16
from utils.eval_utils import logVideos

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BSUV-Net-2.0 pyTorch')
    parser.add_argument('--network', metavar='Network', dest='network', type=str, default='unetvgg16',
                        help='Which network to use. unetvgg16, unet_attention, sparse_unet, R2AttU, SEnDec_cnn_lstm')

    parser.add_argument('--temporal_network', metavar='Temporal network', dest='temporal_network', default='avfeat_confeat_tdr',
                        help='Which temporal network will use. no, avfeat, tdr, avfeat_confeat, avfeat_confeat_tdr, avfeat_tdr')

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
    parser.add_argument('--flux_ch', metavar='Flux tensor', dest='flux_ch', type=int, default=0,
                        help='Whether to use the flux tensor input or not. 0 or 1')
    parser.add_argument('--current_fr', metavar='Current Frame', dest='current_fr', type=int, default=0,
                        help='Whether to use the current frame, 0 or 1')
    parser.add_argument('--patch_frame_size', metavar='Patch frame size', dest='patch_frame_size', type=int, default=0,
                        help='Whether to use the patch frame, last n th frame or not. 0, n: number of last frame')


    # Temporal network parameters
    parser.add_argument('--temporal_history', metavar='Temporal History', dest='temporal_history', type=int,
                        default=50, help='The number of historical frames')
    parser.add_argument('--temporal_kernel_size', metavar='Temporal History', dest='temporal_kernel_size', type=int,
                        default=8, help='Kernel size of temporal network')

    # Cross-validation
    parser.add_argument('--set_number', metavar='Which training-test split to use from config file', dest='set_number',
                        type=int, default=[6], help='Training and test videos will be selected based on the set number')

    # Model name
    parser.add_argument('--model_name', metavar='Name of the model for log keeping', dest='model_name',
                        type=str, default='BSUV-Net 2.0',
                        help='Name of the model to be used in output csv and checkpoints')

    args = parser.parse_args()
    network = args.network
    temporal_network = args.temporal_network
    empty_bg = args.empty_bg
    current_fr = args.current_fr
    patch_frame_size = args.patch_frame_size
    use_flux_tensor = args.flux_ch
    recent_bg = True if args.recent_bg == 1 else False
    seg_ch = True if args.seg_ch == 1 else False
    inp_size = args.inp_size
    if inp_size == 0:
        inp_size = None
    else:
        inp_size = (inp_size, inp_size)

    # Temporal network parameters
    temporal_length = args.temporal_history
    temporal_kernel_size = args.temporal_kernel_size

    use_temporal_network = True if temporal_network != 'no' else False

    set_number = args.set_number
    cuda = True

    # naming for log keeping
    fname = args.model_name + "_network_" + network

    save_dir = data_config.save_dir
    mdl_dir = os.path.join(save_dir, fname)

    # load model
    num_ch_per_inp = (1 + (1 * seg_ch))
    num_inp = (1 * (empty_bg != "no")) + (1 * recent_bg) + 1 * current_fr + (1 * (patch_frame_size != 0))
    num_ch = num_inp * num_ch_per_inp + 1 * (use_flux_tensor == True)

    if network == "unetvgg16":
        model = unet_vgg16(inp_ch=num_ch, kernel_size=3, skip=1, temporal_network=temporal_network,
                           temporal_length=temporal_length, filter_size=temporal_kernel_size)
    elif network.startswith("unet_attention"):
        model = AttU_Net(inp_ch=num_ch, temporal_network=temporal_network, temporal_length=temporal_length,
                         filter_size=temporal_kernel_size)
    elif network.startswith("sparse_unet"):
        model = FgNet(inp_ch=num_ch, temporal_network=temporal_network, temporal_length=temporal_length,
                         filter_size=temporal_kernel_size)
    elif network.startswith('R2AttU'):
        model = R2AttU_Net(inp_ch=num_ch, temporal_network=temporal_network, temporal_length=temporal_length,
                         filter_size=temporal_kernel_size)
    elif network.startswith('SEnDec_cnn_lstm'):
        model = SEnDec_cnn_lstm(inp_ch=num_ch, patch_frame_size=patch_frame_size)

    else:
        raise ValueError(f"network = {network} is not defined")

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
            patch_frame_size=patch_frame_size,
            use_flux_tensor=use_flux_tensor,
            use_temporal_network=use_temporal_network,
            temporal_length=temporal_length,
            recent_bg=recent_bg,
            segmentation_ch=seg_ch,
            save_vid=False,
            use_selected=200,
            set_number=set_number,
            debug=False
        )

    print(f"Saved results to {csv_path}")