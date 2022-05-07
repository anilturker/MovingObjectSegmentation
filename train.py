# Imports

import argparse
import os
import time
import configs.data_config as data_config
import configs.full_cv_config as tr_test_config
import torch
import torch.optim as optim
from utils import augmentations as aug
from utils.data_loader import CDNet2014Loader
from utils import losses
from models.unet import unet_vgg16
from models.unet_3d import UNet_3D
from models.dfr_net import DFR
from models.unet_attention import AttU_Net, R2AttU_Net
from models.convlstm_network import SEnDec_cnn_lstm
from models.sparse_unet import FgNet
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


DEBUG = False


def print_debug(s):
    if DEBUG:
        print(s)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MOS-Net pyTorch')
    parser.add_argument('--network', metavar='Network', dest='network', type=str, default='unetvgg16',
                        help='Which network to use. unetvgg16, unet_attention, unet3d, sparse_unet, '
                             '3dfr, R2AttU, SEnDec_cnn_lstm')

    parser.add_argument('--temporal_network', metavar='Temporal network', dest='temporal_network',
                        default='avfeat_v2',
                        help='Add which temporal network will use(avfeat, avfeat_v2, avfeat_full, '
                             'confeat, fpm, tdr). Otherwise use no')

    # Input images
    parser.add_argument('--inp_size', metavar='Input Size', dest='inp_size', type=int, default=224,
                        help='Size of the inputs. If equals 0, use the original sized images. '
                             'Assumes square sized input')
    parser.add_argument('--use_selected', metavar='Use selected frames', dest='use_selected', type=int, default=200,
                        help='Number of selected frames to be used (0 or 200)')
    parser.add_argument('--empty_bg', metavar='Empty Background Frame', dest='empty_bg', type=str, default='no',
                        help='Which empty background to use? no, manual or automatic')
    parser.add_argument('--recent_bg', metavar='Recent Background Frame', dest='recent_bg', type=int, default=0,
                        help='Use recent background frame as an input as well. 0 or 1')
    parser.add_argument('--seg_ch', metavar='Segmentation', dest='seg_ch', type=int, default=0,
                        help='Whether to use the FPM channel input or not. 0 or 1')
    parser.add_argument('--flux_ch', metavar='Flux tensor', dest='flux_ch', type=int, default=1,
                        help='Whether to use the flux tensor input or not. 0 or 1')
    parser.add_argument('--current_fr', metavar='Current Frame', dest='current_fr', type=int, default=1,
                        help='Whether to use the current frame, 0 or 1')
    parser.add_argument('--patch_frame_size', metavar='Patch frame size', dest='patch_frame_size', type=int, default=1,
                        help='Whether to use the patch frame, last n th frame or not. 0, n: number of last frame'
                             '(not included the current frame)')


    # Temporal network parameters
    parser.add_argument('--temporal_history', metavar='Temporal History', dest='temporal_history', type=int,
                        default=50, help='The number of historical frames')
    parser.add_argument('--temporal_kernel_size', metavar='Temporal History', dest='temporal_kernel_size', type=int,
                        default=8, help='Kernel size of temporal network')


    # Optimization
    parser.add_argument('--lr', metavar='Learning Rate', dest='lr', type=float, default=1e-3,
                        help='learning rate of the optimization')
    parser.add_argument('--weight_decay', metavar='weight_decay', dest='weight_decay', type=float, default=1e-2,
                        help='weight decay of the optimization')
    parser.add_argument('--num_epochs', metavar='Number of epochs', dest='num_epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', metavar='Minibatch size', dest='batch_size', type=int, default=1,
                        help='Number of samples per minibatch')
    parser.add_argument('--loss', metavar='Loss function to be used', dest='loss', type=str,
                        default='jaccard',
                        help='Loss function to be used ce for cross-entropy, focal-loss, tversky-loss'
                             'tversky-bce-loss, focal-tversky-loss or jaccard')
    parser.add_argument('--opt', metavar='Optimizer to be used', dest='opt', type=str, default='adam',
                        help='sgd, rmsprop or adam')

    # Data augmentations
    parser.add_argument('--aug_noise', metavar='Data Augmentation for noise', dest='aug_noise', type=int, default=1,
                        help='Whether to use Data Augmentation for noise. 0 or 1')
    parser.add_argument('--aug_rsc', metavar='Data Augmentation for randomly-shifted crop', dest='aug_rsc', type=int,
                        default=1,
                        help='Whether to use randomly-shifted crop. 0 or 1')
    parser.add_argument('--aug_ptz', metavar='Data Augmentation for PTZ camera crop', dest='aug_ptz', type=int,
                        default=0, help='Whether to use PTZ camera crop 0 or 1')
    parser.add_argument('--aug_id', metavar='Data Augmentation for Illumination Difference', dest='aug_id', type=int,
                        default=1,
                        help='Whether to use Data Augmentation for Illumination Difference. 0 or 1')


    # Checkpoint
    parser.add_argument('--model_chk', metavar='Checkpoint for the model', dest='model_chk', type=int, default=0,
                        help='Whether to use checkpoint, 0 or 1')

    # Cross-validation
    parser.add_argument('--set_number', metavar='Which training-test split to use from config file', dest='set_number',
                        type=int, default=5
                        , help='Training and test videos will be selected based on the set number')

    # Model name
    parser.add_argument('--model_name', metavar='Name of the model for log keeping', dest='model_name',
                        type=str, default='MOS-Net',
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
    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    loss = args.loss
    opt = args.opt
    inp_size = args.inp_size
    use_selected = args.use_selected
    if inp_size == 0:
        inp_size = None
    else:
        inp_size = (inp_size, inp_size)

    # Temporal network parameters
    temporal_length = args.temporal_history
    temporal_kernel_size = args.temporal_kernel_size

    use_temporal_network = True if temporal_network != 'no' or network == '3dfr' else False

    aug_noise = args.aug_noise
    aug_rsc = args.aug_rsc
    aug_ptz = args.aug_ptz
    aug_id = args.aug_id

    model_chk = args.model_chk
    if model_chk == '':
        model_chk = None
    set_number = args.set_number

    cuda = True if torch.cuda.is_available() else False

    # Initializations
    dataset_tr = tr_test_config.datasets_tr[set_number]
    dataset_test = tr_test_config.datasets_test[set_number]
    save_dir = data_config.save_dir

    # naming for log keeping
    fname = args.model_name + "_fusion_net_" + network + "_temporal_net_" + temporal_network + "_" \
            + "inp_selection_" + str(1 * (empty_bg != "no")) + str(1 * recent_bg) + str(1 * seg_ch) \
            + str(1 * use_flux_tensor) + str(1 * current_fr) + "_patch_last_frames_" + str(patch_frame_size)

    print(f"Model started: {fname}")

    # Initialize Tensorboard
    writer = SummaryWriter(f"tb_runs/{fname}")
    print("Initialized TB")

    mdl_dir = os.path.join(save_dir, fname)

    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)

    # Augmentations
    crop_and_aug = []
    if inp_size is not None:
        crop_and_aug = [aug.RandomCrop(inp_size)]

    if aug_rsc and inp_size is not None:
        crop_and_aug.append(aug.RandomJitteredCrop(inp_size))

    if aug_ptz > 0:
        crop_and_aug.append(
            [
                aug.RandomZoomCrop(inp_size),
                aug.RandomPanCrop(inp_size),
            ]
        )

    additional_augs = []

    if aug_id:
        ill_global, std_ill_diff = (0.1, 0.04), (0.1, 0.04)
        additional_augs.append([aug.AdditiveRandomIllumation(ill_global, std_ill_diff)])
    if aug_noise:
        noise = 0.01
        additional_augs.append([aug.AdditiveNoise(noise)])

    mean_rgb = [x for x in [0.485]]
    std_rgb = [x for x in [0.229]]
    mean_seg = [x for x in [0.5]]
    std_seg = [x for x in [0.5]]

    num_ch_per_inp = (1 + (1 * seg_ch))
    num_inp = (1 * (empty_bg != "no")) + (1 * recent_bg) + 1 * current_fr + (1 * (patch_frame_size != 0))
    num_ch = num_inp * num_ch_per_inp + 1 * (use_flux_tensor == True)


    transforms_tr = [
        crop_and_aug,
        *additional_augs,
        [aug.ToTensor()],
        [aug.NormalizeTensor(mean_rgb=mean_rgb, std_rgb=std_rgb,
                             mean_seg=mean_seg, std_seg=std_seg, segmentation_ch=seg_ch,
                             recent_bg=recent_bg, empty_bg=(empty_bg != "no"), current_fr=current_fr,
                             patch_frame_size=patch_frame_size)]
    ]

    transforms_test = [
        [aug.ToTensor()],
        [aug.NormalizeTensor(mean_rgb=mean_rgb, std_rgb=std_rgb,
                             mean_seg=mean_seg, std_seg=std_seg, segmentation_ch=seg_ch,
                             recent_bg=recent_bg, empty_bg=(empty_bg != "no"), current_fr=current_fr,
                             patch_frame_size=patch_frame_size)]
    ]

    dataloader_tr = CDNet2014Loader(
        dataset_tr, empty_bg=empty_bg, current_fr=current_fr, patch_frame_size=patch_frame_size,
        use_flux_tensor=use_flux_tensor, recent_bg=recent_bg,
        use_temporal_network=use_temporal_network, temporal_length=temporal_length, use_selected=use_selected,
        segmentation_ch=seg_ch, transforms=transforms_tr, shuffle=True
    )
    dataloader_test = CDNet2014Loader(
        dataset_test, empty_bg=empty_bg, current_fr=current_fr, patch_frame_size=patch_frame_size,
        use_flux_tensor=use_flux_tensor, recent_bg=recent_bg,
        use_temporal_network=use_temporal_network, temporal_length=temporal_length, use_selected=200,
        segmentation_ch=seg_ch, transforms=transforms_test, shuffle=True
    )

    tensorloader_tr = torch.utils.data.DataLoader(
        dataset=dataloader_tr, batch_size=batch_size, shuffle=True, num_workers=1
    )
    tensorloader_test = torch.utils.data.DataLoader(
        dataset=dataloader_test, batch_size=1, shuffle=True, num_workers=1
    )

    # load model
    if network == "unetvgg16":
        model = unet_vgg16(inp_ch=num_ch, kernel_size=3, skip=1, temporal_network=temporal_network,
                           temporal_length=temporal_length, filter_size=temporal_kernel_size)
    elif network.startswith("unet_attention"):
        model = AttU_Net(inp_ch=num_ch, temporal_network=temporal_network, temporal_length=temporal_length,
                         filter_size=temporal_kernel_size)
    elif network.startswith("sparse_unet"):
        model = FgNet(inp_ch=num_ch, temporal_network=temporal_network, temporal_length=temporal_length,
                         filter_size=temporal_kernel_size)
    elif network == "3dfr":
        model = DFR(inp_ch=num_ch, kernel_size=3, skip=1, temporal_length=temporal_length,
                    filter_size=temporal_kernel_size)
    elif network.startswith('R2AttU'):
        model = R2AttU_Net(inp_ch=num_ch, temporal_network=temporal_network, temporal_length=temporal_length,
                         filter_size=temporal_kernel_size)
    elif network.startswith('SEnDec_cnn_lstm'):
        model = SEnDec_cnn_lstm(inp_ch=1)  # inp_ch is 3 if rgb input else 1
    elif network.startswith('unet3d'):
        model = UNet_3D(inp_ch=1)  # inp_ch is 3 if rgb input else 1

    else:
        raise ValueError(f"network = {network} is not defined")

    for p in model.parameters():
        p.requires_grad = True

    # setup optimizer
    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    else:
        raise ("opt=%s is not defined, please choose from ('adam', 'sgd')." % opt)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2,
                                                     factor=0.5, min_lr=1e-04, verbose=True)

    if loss == "jaccard":
        loss_func = losses.jaccard_loss
    elif loss == "ce":
        loss_func = losses.binary_cross_entropy_loss
    elif loss == "focal-loss":
        loss_func = losses.binary_focal_loss
    elif loss == "tversky-loss":
        loss_func = losses.tverskyLoss
    elif loss == "tversky-bce-loss":
        loss_func = losses.tverskyLoss_bce_loss
    elif loss == "focal-tversky-loss":
        loss_func = losses.focal_tversky_loss
    else:
        raise ("loss=%s is not defined, please choose from ('jaccard', 'ce', 'focal-loss', 'tversky-loss, '"
               " focal-tversky-loss).")

    # Print model's state_dict
    print_debug("Model's state_dict:")
    for param_tensor in model.state_dict():
        print_debug(param_tensor + "\t" + str(model.state_dict()[param_tensor].size()))

    if cuda:
        model = model.cuda()

    best_f = 0.0
    start_epoch = 0
    if model_chk:
        chk_path = f"{mdl_dir}/checkpoint.pth"
        if os.path.exists(chk_path):
            assert os.path.isfile(chk_path), f"No checkpoint is found in {chk_path}"
            print(f"=> loading checkpoint '{model_chk}'")
            checkpoint = torch.load(chk_path)
            start_epoch = checkpoint['epoch']
            best_f = checkpoint['best_f']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.param_groups[0]["lr"] = lr
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_chk, checkpoint['epoch']))


    # training
    epoch_loss = 0.0 # For saving the best model
    epoch_acc = 0.0
    epoch_f = 0.0

    st = time.time()
    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times
        for phase, tensorloader in [("Train", tensorloader_tr), ("Test", tensorloader_test)]:
            running_loss, running_acc, running_f = 0.0, 0.0, 0.0
            tp, fp, fn = 0, 0, 0
            if phase == "Train":
                optimizer.zero_grad()
            for i, data in enumerate(tensorloader):
                if phase == "Train":
                    model.train()
                else:
                    model.eval()

                if phase == "Train":
                    # get the inputs; data is a list of [inputs, labels]
                    if use_temporal_network is False:
                        inputs, labels = data[0].float(), data[1].float()
                    else:
                        inputs, labels = torch.cat((data[0], data[1]), dim=1), data[2].float()
                    if cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    labels_1d, outputs_1d = losses.getValid(labels, outputs)
                    loss = loss_func(labels_1d, outputs_1d)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with torch.no_grad():
                        # get the inputs; data is a list of [inputs, labels]
                        if use_temporal_network is False:
                            inputs, labels = data[0].float(), data[1].float()
                        else:
                            inputs, labels = torch.cat((data[0], data[1]), dim=1), data[2].float()
                        if cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()
                        outputs = model(inputs)
                        labels_1d, outputs_1d = losses.getValid(labels, outputs)
                        loss = loss_func(labels_1d, outputs_1d)

                # print statistics
                running_loss += loss.item()

                tp += torch.sum(labels_1d * outputs_1d).item()
                fp += torch.sum((1 - labels_1d) * outputs_1d).item()
                fn += torch.sum(labels_1d * (1 - outputs_1d)).item()

                del inputs, labels, outputs, labels_1d, outputs_1d
                if (i + 1) % 100 == 0:  # print every 2000 mini-batches
                    # Calculate the statistics
                    prec = tp / (tp + fp) if tp + fp > 0 else float('nan')
                    recall = tp / (tp + fn) if tp + fn > 0 else float('nan')
                    f_score = 2 * (prec * recall) / (prec + recall) if prec + recall > 0 else float('nan')

                    print("::%s::[%d, %5d] loss: %.3f, prec: %.3f, recall: %.3f, f_score: %.3f" %
                          (phase, epoch + 1, i + 1,
                           running_loss / (i + 1), prec, recall, f_score))

            epoch_loss = running_loss / len(tensorloader)

            # Calculate the statistics
            prec = tp / (tp + fp) if tp + fp > 0 else float('nan')
            recall = tp / (tp + fn) if tp + fn > 0 else float('nan')
            f_score = 2 * (prec * recall) / (prec + recall) if prec + recall > 0 else float('nan')

            print("::%s:: Epoch %d loss: %.3f, prec: %.3f, recall: %.3f, f_score: %.3f, lr : %.6f, elapsed time: %s"\
                  % (phase, epoch + 1, epoch_loss, prec, recall, f_score,  optimizer.param_groups[0]["lr"],
                     (time.time() - st)))

            writer.add_scalar(f"{phase}/loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase}/prec", prec, epoch)
            writer.add_scalar(f"{phase}/recall", recall, epoch)
            writer.add_scalar(f"{phase}/f", f_score, epoch)

            if phase.startswith("Test"):
                if f_score > best_f:
                    print("The model weights that gives best f1 score on the test data are saving")
                    best_f = f_score
                    torch.save(model, f"{mdl_dir}/model_best.mdl")

            # Learning rate scheduler
            if phase.startswith("Train"):
                scheduler.step(epoch_loss)

            # Save the checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "best_f": best_f,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            torch.save(checkpoint, f"{mdl_dir}/checkpoint.pth")
            if (epoch + 1) % 20 == 0:
                torch.save(model, f"{mdl_dir}/model_epoch{epoch + 1}.mdl")

            writer.add_hparams({'network': network, 'temporal_network': temporal_network, 'current_epoch': epoch + 1,
                                'batch_size': batch_size, 'empty_bg': empty_bg, 'current_fr': current_fr,
                                'use_flux_tensor': use_flux_tensor, 'recent_bg': recent_bg, 'seg_ch': seg_ch,
                                'lr': lr, 'weight_decay': weight_decay, 'loss': args.loss,
                                'temporal_length': temporal_length,
                                'temporal_kernel_size': temporal_kernel_size, 'opt': opt, 'inp_size': inp_size},
                               {'epoch_loss': epoch_loss, 'epoch_prec': prec, 'epoch_recall': recall,
                                'epoch_f': f_score})

            st = time.time()

    # save the last model
    torch.save(model, f"{mdl_dir}/model_last.mdl")

    temporal_length = args.temporal_history
    temporal_kernel_size = args.temporal_kernel_size

    print('Finished Training')
