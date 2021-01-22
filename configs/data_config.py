"""
Configuration file for dataset paths
"""

root_dir = "./dataset"
save_dir = "./saved_models"

# Paths for the CDNet2014 images, their background frames, segmentation maps and ground truths
current_fr_dir = "{}/currentFr/{{cat}}/{{vid}}/input".format(root_dir)
current_fr_path = "{}/currentFr/{{cat}}/{{vid}}/input/in{{fr_id}}.jpg".format(root_dir)
gt_path = "{}/currentFr/{{cat}}/{{vid}}/groundtruth/gt{{fr_id}}.png".format(root_dir)
empty_bg_root = "{}/emptyBg/{{cat}}/{{vid}}".format(root_dir)
empty_bg_path = "{}/emptyBg/{{cat}}/{{vid}}/empty_bg{{fr_id}}.jpg".format(root_dir)
recent_bg_path = "{}/recentBg/{{cat}}/{{vid}}/recent_bg{{fr_id}}.jpg".format(root_dir)

current_fr_fpm_path = "{}/currentFrFpm/{{cat}}/{{vid}}/fpm{{fr_id}}.jpg".format(root_dir)
empty_bg_fpm_path = "{}/emptyBgFpm/{{cat}}/{{vid}}/fpm{{fr_id}}.jpg".format(root_dir)
recent_bg_fpm_path = "{}/recentBgFpm/{{cat}}/{{vid}}/fpm{{fr_id}}.jpg".format(root_dir)

# Directory for the selected background frames
selected_frs_200_csv = "{}/CDNET2014_selected_frames_200.csv".format(root_dir)

# Path to the temp roi file
temp_roi_path = "{}/currentFr/{{cat}}/{{vid}}/temporalROI.txt".format(root_dir)

# Locations of each video in the CSV file
csv_header2loc = {'len': 160, 'highway': 1, 'pedestrians': 4, 'office': 7, 'PETS2006': 10, 'badminton': 13, 'traffic': 16,
                  'boulevard': 19, 'sidewalk': 22, 'skating': 25, 'blizzard': 28, 'snowFall': 31, 'wetSnow': 34, 'boats': 37,
                  'canoe': 40, 'fall': 43, 'fountain01': 46, 'fountain02': 49, 'overpass': 52, 'abandonedBox': 55,
                  'parking': 58, 'sofa': 61, 'streetLight': 64, 'tramstop': 67, 'winterDriveway': 70,
                  'port_0_17fps': 73, 'tramCrossroad_1fps': 76, 'tunnelExit_0_35fps': 79, 'turnpike_0_5fps': 82,
                  'bridgeEntry': 85, 'busyBoulvard': 88, 'fluidHighway': 91, 'streetCornerAtNight': 94, 'tramStation': 97,
                  'winterStreet': 100, 'continuousPan': 103, 'intermittentPan': 106, 'twoPositionPTZCam': 109,
                  'zoomInZoomOut': 112, 'backdoor': 115, 'bungalows': 118, 'busStation': 121, 'copyMachine': 124,
                  'cubicle': 127, 'peopleInShade': 130, 'corridor': 133, 'diningRoom': 136, 'lakeSide': 139, 'library': 142,
                  'park': 145, 'turbulence0': 148, 'turbulence1': 151, 'turbulence2': 154, 'turbulence3': 157}