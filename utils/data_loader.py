import csv
import os
import numpy as np
import random
import torch.utils.data as data
from configs import data_config
import cv2
import glob

class CDNet2014Loader(data.Dataset):
    """
    Data loader class
    :param input_tuples ([(str, str, int)]): List of input tuples with category, video and frame number
    :param catvid_to_bg_ids (dict {str:[int]}): Dicitonary object with "{}/{}".format(category, video)
                                                as keys and list of background frame ids as values
    """

    def __init__(self, dataset, empty_bg="no", current_fr=1, use_flux_tensor=False, empty_win_len=0, recent_bg=False,
                 patch_frame_size=0, segmentation_ch=False, use_temporal_network=False, temporal_length = 50,
                 use_selected=None, transforms=None, multiplier=16, shuffle=False):
        """Initialization of data loader
        Args:
            :dataset (dict):                Dictionary of dataset. Keys are the categories (string),
                                            values are the arrays of video names (strings).
            :empty_bg (str):                'no': no empty background
                                            'manual': manually created empty background
                                            'automatic' : median of first k frames as empty background
            :current_fr                     '0': no current frame
                                            '1': use current frame
            :use_flux_tensor                'False' or 'True'
            :empty_win_length (int):        Number of initial frames for the median operation fro creating an empty
                                            background. Only used when empty_bg='automatic'.
                                            0 means median of all of the frames in the video
            :recent_bg (boolean):           Boolean for using the recent background frame
            :patch_frame_size               The size of patch frame contains last n frame
            :segmentation_ch (boolean):     Boolean for using the segmentation maps
            :use_temporal_network(boolean): Boolean for using temporal network(AvFeat)
            :temporal_length(int):          Number of frames for temporal network(AvFeat)
            :use_selected (int):            Number of selected frames to be used (None or 200)
            :transforms (torchvision.transforms):   Transforms to be applied to each input
            :multiplier (int):              Clip the outputs to be a multiple of multiplier. If 0 -> no clipping
            :shuffle (boolean):             Return shuffled elements with no end item
        """

        if use_selected and use_selected != -1:
            if use_selected == 200:
                selected_frs_csv = data_config.selected_frs_200_csv
            else:
                raise(f"Number of selected frames can be None or 200 but {use_selected} given")

            with open(selected_frs_csv) as f:
                reader = csv.reader(f)
                selected_frs = list(reader)

            # Create a dictionary of cat/vid -> list of selected frames
            catvid_to_selected_frs = {arr[0]:list(map(int, arr[1].split())) for arr in selected_frs}

        input_tuples = []
        for cat, vid_arr in dataset.items():
            for vid in vid_arr:

                # Read temporal roi file
                roi_path = data_config.temp_roi_path.format(cat=cat, vid=vid)
                with open(roi_path) as f:
                    reader = csv.reader(f)
                    temp_roi = list(reader)
                temp_roi = list(map(int, temp_roi[0][0].split()))

                # Find out the required frame ids (either selected or all the ones that have gt)
                if use_selected == -1:
                    last_fr = int(sorted(glob.glob(os.path.join(data_config.current_fr_dir.format(cat=cat, vid=vid),
                                                                "*.jpg")))[-1][-10:-4])
                    fr_ids = [idx for idx in range(1, last_fr+1)]
                elif use_selected:
                    fr_ids = catvid_to_selected_frs[f"{cat}/{vid}"]
                else:
                    fr_ids = [idx for idx in range(temp_roi[0], temp_roi[1]+1)]

                if use_temporal_network:
                    first_proper_id = temp_roi[0] + temporal_length - 1
                    for fr_id in fr_ids:
                        if first_proper_id > fr_id:
                            fr_ids.remove(fr_id)

                for fr_id in fr_ids:
                        input_tuples.append((cat, vid, fr_id))

        if shuffle:
            random.shuffle(input_tuples)

        self.input_tuples = input_tuples
        self.n_data = len(input_tuples)
        self.empty_bg = empty_bg
        self.current_fr = current_fr
        self.patch_frame_size = patch_frame_size
        self.use_flux_tensor = use_flux_tensor
        self.empty_win_len = empty_win_len
        self.recent_bg = recent_bg
        self.segmentation_ch = segmentation_ch
        self.transforms = transforms
        self.multiplier = multiplier
        self.shuffle = shuffle
        self.use_temporal_network = use_temporal_network
        self.temporal_length = temporal_length

    def __getitem__(self, item):

        cat, vid, fr_id = self.input_tuples[item]

        # Construct the input
        inp = {"current_fr":None, "current_fr_seg":None,
               "empty_bg_seg":None, "empty_bg":None,
               "recent_bg_seg":None, "recent_bg":None}

        if self.empty_bg == "manual":
            empty_bg_path = data_config.empty_bg_path.format(
                cat=cat, vid=vid, fr_id=str(fr_id).zfill(6))
            empty_bg_fpm_path = data_config.empty_bg_fpm_path.format(
                cat=cat, vid=vid, fr_id=str(fr_id).zfill(6))
            if not os.path.exists(empty_bg_path):
                empty_bg_id = random.choice(
                    os.listdir(
                        data_config.empty_bg_root.format(cat=cat, vid=vid)
                    ))[-10:-4]
                empty_bg_path = data_config.empty_bg_path.format(
                    cat=cat, vid=vid, fr_id=empty_bg_id)
                empty_bg_fpm_path = data_config.empty_bg_fpm_path.format(
                    cat=cat, vid=vid, fr_id=empty_bg_id)
            if not os.path.exists(empty_bg_path):
                raise(f"No empty BG found for {cat}/{vid}")

            inp["empty_bg"] = self.__readGray(empty_bg_path)

        elif self.empty_bg != "no":
            raise ValueError(f"empty_bg should be no or manual; but given '{self.empty_bg}'")

        if self.segmentation_ch and self.empty_bg != "no":
            inp["empty_bg_seg"] = self.__readGray(empty_bg_fpm_path)

        if self.current_fr:
            inp["current_fr"] = self.__readGray(data_config.current_fr_path\
                                .format(cat=cat, vid=vid, fr_id=str(fr_id).zfill(6)))

        if self.patch_frame_size:
            for i, id in enumerate(range(fr_id - self.patch_frame_size + 1, fr_id + 1)):
                temporal_frame = self.__readGray(data_config.current_fr_path\
                                    .format(cat=cat, vid=vid, fr_id=str(id).zfill(6)))

                inp["patch_frame_" + str(i)] = temporal_frame

        if self.segmentation_ch and self.recent_bg:
            inp["recent_bg_seg"] = self.__readGray(data_config.recent_bg_fpm_path\
                                       .format(cat=cat, vid=vid, fr_id=str(fr_id).zfill(6)))

        if self.recent_bg:
            inp["recent_bg"] = self.__readGray(data_config.recent_bg_path\
                                   .format(cat=cat, vid=vid, fr_id=str(fr_id).zfill(6)))
        if self.segmentation_ch:
            inp["current_fr_seg"] = self.__readGray(data_config.current_fr_fpm_path\
                                        .format(cat=cat, vid=vid, fr_id=str(fr_id).zfill(6)))

        label = self.__readGray(data_config.gt_path \
                                .format(cat=cat, vid=vid, fr_id=str(fr_id).zfill(6)))

        if self.use_flux_tensor:
            inp["flux_tensor"] = self.__readGray(data_config.flux_tensor_path\
                                                 .format(cat=cat, vid=vid, fr_id=str(fr_id).zfill(6)))

        if self.patch_frame_size:
            for i, id in enumerate(range(fr_id - self.patch_frame_size + 1, fr_id + 1)):
                temporal_frame = self.__readGray(data_config.current_fr_path\
                                    .format(cat=cat, vid=vid, fr_id=str(id).zfill(6)))

                inp["patch_frame_" + str(i)] = temporal_frame

        if self.use_temporal_network:
            for i, id in enumerate(range(fr_id-self.temporal_length+1, fr_id+1)):
                temporal_frame = self.__readGray(data_config.current_fr_path\
                                .format(cat=cat, vid=vid, fr_id=str(id).zfill(6)))

                inp["temporal_patch_" + str(i)] = temporal_frame

        # Apply transform and clipping
        for transform_arr in self.transforms:
            if len(transform_arr) > 0:
                inp, label = self.__selectrandom(transform_arr)(inp, label)

        if self.multiplier > 0:
            c, h, w = label.shape
            h_valid = int(h/self.multiplier)*self.multiplier
            w_valid = int(w/self.multiplier)*self.multiplier
            inp, label = inp[:, :h_valid, :w_valid], label[:, :h_valid, :w_valid]

        # reformat label such that FG=1, BG=0, everything else = -1
        label[label <= 0.05] = 0 # BG
        label[np.abs(label-0.5) < 0.45] = -1
        label[label >= 0.95] = 1 # FG

        if self.use_temporal_network:
            temporal_patch_first_index = inp.shape[0] - self.temporal_length
            temp_patch_inp = inp[temporal_patch_first_index:]
            return inp[:temporal_patch_first_index], temp_patch_inp, label
        else:
            return inp, label

    def __len__(self):
        return self.n_data

    def __readRGB(self, path):
        assert os.path.exists(path), f"{path} does not exist"
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float)/255

    def __readGray(self, path):
        assert os.path.exists(path), f"{path} does not exist"
        return np.expand_dims(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), -1).astype(np.float)/255

    def __selectrandom(self, arr):
        choice = arr.copy()
        while isinstance(choice, list):
            choice = random.choice(choice)
        return choice

