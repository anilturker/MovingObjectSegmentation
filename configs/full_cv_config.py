"""Cross-validation sets used for video-agnostic evaluation
"""
datasets_tr = {
    0: {'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
        'cameraJitter':['badminton', 'traffic', 'boulevard', 'sidewalk'],
        'badWeather':['skating', 'blizzard', 'snowFall', 'wetSnow'],
        'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
        'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
        'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
        'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
        'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
        'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
        'thermal':['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
        'turbulence':['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3']
        }, # Full dataset
    1: {'baseline':['pedestrians', 'office', 'PETS2006'],
        'cameraJitter':['traffic', 'boulevard', 'sidewalk'],
        'badWeather':['skating', 'snowFall', 'wetSnow'],
        'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01'],
        'intermittentObjectMotion':['abandonedBox', 'parking', 'streetLight', 'tramstop', 'winterDriveway'],
        'lowFramerate':['tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
        'nightVideos':['fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
        'PTZ':['intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
        'shadow':['backdoor', 'bungalows', 'busStation', 'cubicle', 'peopleInShade'],
        'thermal':['diningRoom', 'library', 'park'],
        'turbulence':['turbulence1', 'turbulence2', 'turbulence3']
        }, # CV fold 1
    2: {'baseline':['highway', 'office', 'PETS2006'],
        'cameraJitter':['badminton', 'boulevard', 'sidewalk'],
        'badWeather':['blizzard', 'snowFall', 'wetSnow'],
        'dynamicBackground':['canoe', 'fall', 'fountain02', 'overpass'],
        'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop'],
        'lowFramerate':['port_0_17fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
        'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight'],
        'PTZ':['continuousPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
        'shadow':['backdoor', 'bungalows', 'copyMachine', 'cubicle', 'peopleInShade'],
        'thermal':['corridor', 'diningRoom', 'lakeSide', 'park'],
        'turbulence':['turbulence0', 'turbulence2', 'turbulence3']
        }, # CV fold 2
    3: {'baseline':['highway', 'pedestrians', 'PETS2006'],
        'cameraJitter':['badminton', 'traffic', 'sidewalk'],
        'badWeather':['skating', 'blizzard', 'snowFall'],
        'dynamicBackground':['boats', 'fall', 'fountain01', 'fountain02', 'overpass'],
        'intermittentObjectMotion':['sofa', 'streetLight', 'tramstop', 'winterDriveway'],
        'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'turnpike_0_5fps'],
        'nightVideos':['bridgeEntry', 'busyBoulvard', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
        'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam'],
        'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine'],
        'thermal':['corridor', 'lakeSide', 'library', 'park'],
        'turbulence':['turbulence0', 'turbulence1', 'turbulence3']
        }, # CV fold 3
    4: {'baseline':['highway', 'pedestrians', 'office'],
        'cameraJitter':['badminton', 'traffic', 'boulevard'],
        'badWeather':['skating', 'blizzard', 'wetSnow'],
        'dynamicBackground':['boats', 'canoe', 'fountain01', 'fountain02', 'overpass'],
        'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'winterDriveway'],
        'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps'],
        'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'tramStation', 'winterStreet'],
        'PTZ':['continuousPan', 'intermittentPan', 'zoomInZoomOut'],
        'shadow':['busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
        'thermal':['corridor', 'diningRoom', 'lakeSide', 'library'],
        'turbulence':['turbulence0', 'turbulence1', 'turbulence2']
        }, # CV fold 4
    5: {
        'lowFramerate':['port_0_17fps'],
        'intermittentObjectMotion':['streetLight']
        }, # small dataset for quick debugging
}

datasets_test = {
    1: {'baseline':['highway'],
        'cameraJitter':['badminton'],
        'badWeather':[ 'blizzard'],
        'dynamicBackground':['fountain02', 'overpass'],
        'intermittentObjectMotion':['sofa'],
        'lowFramerate':['port_0_17fps'],
        'nightVideos':['bridgeEntry', 'busyBoulvard'],
        'PTZ':['continuousPan'],
        'shadow':['copyMachine'],
        'thermal':['corridor', 'lakeSide',],
        'turbulence':['turbulence0']
        },
    2: {'baseline':['pedestrians'],
        'cameraJitter':['traffic'],
        'badWeather':['skating'],
        'dynamicBackground':['boats', 'fountain01'],
        'intermittentObjectMotion':['winterDriveway'],
        'lowFramerate':['tramCrossroad_1fps'],
        'nightVideos':['tramStation', 'winterStreet'],
        'PTZ':['intermittentPan'],
        'shadow':['busStation'],
        'thermal':['library'],
        'turbulence':['turbulence1']
        },
    3: {'baseline':['office'],
        'cameraJitter':['boulevard'],
        'badWeather':['wetSnow'],
        'dynamicBackground':['canoe'],
        'intermittentObjectMotion':['abandonedBox', 'parking'],
        'lowFramerate':['tunnelExit_0_35fps'],
        'nightVideos':['fluidHighway'],
        'PTZ':['zoomInZoomOut'],
        'shadow':['cubicle', 'peopleInShade'],
        'thermal':['diningRoom'],
        'turbulence':['turbulence2']
        },
    4: {'baseline':['PETS2006'],
        'cameraJitter':['sidewalk'],
        'badWeather':['snowFall'],
        'dynamicBackground':['fall'],
        'intermittentObjectMotion':['streetLight', 'tramstop'],
        'lowFramerate':['turnpike_0_5fps'],
        'nightVideos':['streetCornerAtNight'],
        'PTZ':['twoPositionPTZCam'],
        'shadow':['backdoor', 'bungalows'],
        'thermal':['park'],
        'turbulence':['turbulence3']
        },
    5: {
         'lowFramerate':['tramCrossroad_1fps']
        }, # small dataset for quick debugging
}
