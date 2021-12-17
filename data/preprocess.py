from _typeshed import BytesPath
import torch
import mmcv
import numpy as np
import cv2
import os
from mmskeleton.apis.estimation import init_pose_estimator, inference_pose_estimator
from multiprocessing import current_process, Process, Manager
from mmskeleton.utils import cache_checkpoint, third_party
from mmcv.utils import ProgressBar


# def get_all_files(path):
#     allfile = []
#     for dirpath, dirnames, filenames in os.walk(path):
#         for dir in dirnames:
#             allfile.append(os.path.join(dirpath, dir))
#         for name in filenames:
#             allfile.append(os.path.join(dirpath, name))
#     return allfile

# def remove_invalid_player(joint_preds, bboxes):
#     res = 0
#     resSpace = 1000000
#     for i, bbox in enumerate(bboxes):
#         # TODO: remove person who are not in the court

#         # we only need max person
#         bspace = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
#         if i != res and bspace < resSpace:
#             res = i
#             resSpace = bspace
#     return joint_preds[res]

def get_data(detection_cfg,
              estimation_cfg,
              video_dir,
              gpus=1,
              worker_per_gpu=1,
              save_dir=None):
    '''
    transfer .mp4 to .npy, and store data to '/resource'
    npy: m frames * n joints * (x, y)
    :param video_dir:
    :param gpus: 1
    :param worker_per_gpu: 1
    :return:
    '''
    if not third_party.is_exist('mmdet'):
        print(
            '\nERROR: This demo requires mmdet for detecting bounding boxes of person. Please install mmdet first.'
        )
        exit(1)

    joint_data = []
    labels = []
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # all_files = get_all_files(video_dir)
    all_files = ['ff1', 'ff2']
    for video_file in all_files:
        video_frames = mmcv.VideoReader(video_file)
        print(len(video_frames))
        video_name = video_file.strip('/n').split('/')[-1]
        video_name = video_name.split('.')[0]
        all_result = []
        # labels = [action, camera, index] eg. ff_a_01
        labels = video_name.split('_')

        # case for single process
        if gpus == 1 and worker_per_gpu == 1:
            model = init_pose_estimator(detection_cfg, estimation_cfg, device=0)
            prog_bar = ProgressBar(len(video_frames))
            for i, image in enumerate(video_frames):
                # get person-recognition result
                res = inference_pose_estimator(model, image)
                # jp = remove_invalid_player(res['joint_preds'], res['person_bbox'])
                joint_pred = [res['joint_preds'], i]

                all_result.append(joint_pred)
                prog_bar.update()
           
        # sort results, all_result = m frames * n joints * (x, y)
        all_result = sorted(all_result, key=lambda x: x[1])
        all_result = [x[0] for x in all_result]
        print("all_result: ", len(all_result))

        if (len(all_result) == len(video_frames)) and (save_dir is not None):
            print("\nGenerate ", video_name, ".npy")
            video_npy_path = os.path.join(save_dir, video_name+".npy")
            np.save(video_npy_path, all_result)
            print(video_npy_path, " stored")
    
    # test we have the correct npy data
    # result_files = get_all_files(save_dir)
    # for i, file in enumerate(result_files):
    #     data = np.load(file)
    #     print(i, data.shape)
