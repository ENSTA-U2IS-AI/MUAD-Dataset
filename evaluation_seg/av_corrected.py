import json
import os
import sys

import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.utils.data as data


class dataset(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    datasetClass = namedtuple('datasetClass', ['name', 'id', 'train_id', 'object_id', 'instance', 'road_layer'])
    classes = [
        datasetClass('bots',                                        0, 0, (50, 0, 150), False, False),
        datasetClass('traffic_lights_head',                         1, 6, (0, 0, 255), True, False),
        datasetClass('mailbox',                                     2, 255, (200, 50, 50), False, False),
        datasetClass('tram_tracks',                                 3, 0, (130, 90, 255), False, False),
        datasetClass('cyclist',                                     4, 12, (250, 50, 255), True, False),
        datasetClass('traffic_lights_bulb_yellow',                  5, 6, (0, 0, 253), False, False),
        datasetClass('barbacue',                                    6, 255, (200, 60, 200), False, False),
        datasetClass('vase',                                        7, 255, (145, 190, 25), False, False),
        datasetClass('poster',                                      8, 255, (90, 0, 90), False, False),
        datasetClass('yellow_barrel',                               9, 255, (255, 0, 50), False, False),
        datasetClass('vegetation',                                  10, 8, (0, 255, 255), False, False),
        datasetClass('vegetationv2',                                10, 8, (227, 33, 77), False, False),
        datasetClass('motorcycle',                                  11, 17, (50, 100, 100), True, False),
        datasetClass('house',                                       12, 2, (130, 0, 0), False, False),
        datasetClass('construction_helmet_green',                   13, 255, (0, 80, 205), True, False),
        datasetClass('electric_post_insulator_break',               14, 255, (100, 0, 150), False, False),
        datasetClass('uninteresting',                               15, 255, (1, 1, 1), False, False),
        datasetClass('wood',                                        16, 255, (201, 100, 0), False, False),
        datasetClass('construction_scaffold',                       17, 255, (112, 132, 112), False, False),
        datasetClass('car',                                         18, 13, (100, 75, 0), True, False),
        datasetClass('hammock',                                     19, 255, (50, 125, 100), False, False),
        datasetClass('construction_cord',                           20, 255, (22, 82, 152), False, False),
        datasetClass('rock',                                        21, 255, (255, 255, 150), False, False), ### NEW OOD
        datasetClass('suitcase',                                    22, 255, (210, 135, 246), True, False),
        datasetClass('construction_fence',                          23, 4, (50, 80, 50), False, False),
        datasetClass('garbage_bag',                                 24, 20, (130, 130, 130), False, False), ### NEW OOD
        datasetClass('bicycle',                                     25, 18, (150, 10, 60), True, False),
        datasetClass('tree_pit',                                    26, 9, (0, 255, 100), False, False),
        datasetClass('electric_power',                              27, 255, (50, 110, 10), False, False),
        datasetClass('construction_helmet_white',                   28, 255, (0, 80, 201), True, False),
        datasetClass('wind_direction',                              29, 255, (100, 10, 100), False, False),
        datasetClass('safety_vest_03_pink',                         30, 255, (0, 81, 103), True, False),
        datasetClass('segway',                                      31, 255, (20, 80, 50), False, False),
        datasetClass('lane_bike',                                   32, 0, (200, 150, 250), False, False),
        datasetClass('crosswalk',                                   33, 0, (150, 255, 255), False, True),
        datasetClass('baby_cart',                                   34, 255, (210, 130, 240), True, False),
        datasetClass('pile_of_sand',                                35, 255, (35, 100, 40), False, False),
        datasetClass('traffic_signs_poles_or_structure',            36, 5, (255, 150, 0), False, False),
        datasetClass('safety_vest_03_yellow',                       37, 255, (0, 80, 103), True, False),
        datasetClass('parking_area',                                38, 0, (0, 50, 255), False, False),
        datasetClass('dog',                                         39, 255, (200, 150, 255), True, False),
        datasetClass('wheel_chair',                                 40, 255, (210, 135, 247), True, False),
        datasetClass('safety_vest_03_green',                        41, 255, (0, 85, 103), True, False),
        datasetClass('garbage_road',                                42, 0, (50, 50, 51), False, True),
        datasetClass('concrete_benchs',                             43, 255, (150, 10, 110), False, False),
        datasetClass('umbrella_garden',                             44, 255, (200, 180, 20), False, False),
        datasetClass('bus',                                         45, 15, (100, 100, 150), True, False),
        datasetClass('safety_vest_04',                              46, 255, (0, 80, 104), True, False),
        datasetClass('pool',                                        47, 255, (140, 200, 170), False, False),
        datasetClass('building0',                                    48, 2, (150, 0, 250), False, False),
        datasetClass('building',                                    48, 2, (150, 0, 255), False, False),
        datasetClass('buildingv2',                                   48, 2, (54, 48, 163), False, False),
        datasetClass('buildingv3',                                    48, 2, (96, 111, 44), False, False),
        datasetClass('buildingv4',                                   48, 2, (154, 239, 54), False, False),
        datasetClass('buildingv5',                                   48, 2, (148, 249, 102), False, False),
        datasetClass('buildingv6',                                   48, 2, (183, 83, 159), False, False),
        datasetClass('truck',                                       49, 14, (50, 100, 200), True, False),
        datasetClass('road_lines',                                  50, 0, (50, 200, 10), False, True),
        datasetClass('lamp',                                        51, 5, (30, 190, 100), True, False),
        datasetClass('water',                                       52, 255, (130, 160, 210), False, False),
        datasetClass('wall',                                        53, 3, (230, 230, 230), False, False),
        datasetClass('portable_bathroom',                           54, 255, (155, 155, 250), False, False),
        datasetClass('construction_post_cone',                      55, 255, (110, 130, 110), False, False),
        datasetClass('armchair',                                    56, 255, (20, 180, 44), False, False),
        datasetClass('ego_car',                                     57, 13, (5, 5, 5), False, False),
        datasetClass('kerb_stone',                                  58, 1, (20, 40, 80), False, False),
        datasetClass('air_conditioning',                            59, 255, (98, 110, 10), False, False),
        datasetClass('safety_vest_03_red',                          60, 255, (0, 83, 103), True, False),
        datasetClass('press_box',                                   61, 255, (90, 150, 230), False, False),
        datasetClass('biker',                                       62, 12, (250, 100, 255), True, False),
        datasetClass('table',                                       63, 255, (10, 255, 140), False, False),
        datasetClass('tv_antenna',                                  64, 2, (100, 10, 200), False, False),
        datasetClass('beacon_light',                                65, 255, (200, 10, 100), False, False),
        datasetClass('parking_bicycles',                            66, 4, (0, 0, 100), False, False),
        datasetClass('longitudinal_crack',                          67, 255, (213, 128, 0), True, True),
        datasetClass('vending_machine',                             68, 255, (31, 31, 181), False, False),
        datasetClass('tricycle',                                    69, 18, (216, 0, 89), False, False),
        datasetClass('walker',                                      70, 11, (210, 135, 245), True, False),
        datasetClass('chair',                                       71, 255, (5, 95, 10), False, False),
        datasetClass('safety_vest_03_turquoise',                    72, 255, (0, 86, 103), True, False),
        datasetClass('swing',                                       73, 255, (130, 70, 240), False, False),
        datasetClass('electric_post_conductor',                     74, 255, (116, 116, 116), False, False),
        datasetClass('construction_helmet_blue',                    75, 255, (0, 80, 204), True, False),
        datasetClass('umbrella',                                    76, 255, (210, 135, 248), True, False),
        datasetClass('ball',                                        77, 255, (210, 135, 249), True, False),
        datasetClass('sidewalk',                                    78, 1, (150, 200, 130), False, False),
        datasetClass('construction_helmet_red',                     79, 255, (0, 80, 203), True, False),
        datasetClass('jersey_barrier',                              80, 255, (255, 150, 255), False, False),
        datasetClass('traffic_signs',                               81, 7, (255, 0, 255), False, False),
        datasetClass('traffic_signsv2',                               81, 7, (127, 76, 224), False, False),
        datasetClass('terrace',                                     82, 255, (2, 10, 13), False, False),
        datasetClass('container',                                   83, 255, (255, 255, 200), False, False),
        datasetClass('transversal_crack',                           84, 255, (213, 213, 0), True, False),
        datasetClass('kerb_rising_edge',                            85, 1, (20, 40, 90), False, False),
        datasetClass('construction_contrainer',                     86, 255, (113, 133, 113), False, False),
        datasetClass('construction_concrete',                       87, 0, (20, 80, 150), False, False),
        datasetClass('train',                                       88, 16, (30, 80, 230), True, False),
        datasetClass('dog_house',                                   89, 2, (50, 110, 240), False, False),
        datasetClass('water_tank',                                  90, 255, (255, 100, 50), False, False),
        datasetClass('painting',                                    91, 255, (45, 190, 240), False, False),
        datasetClass('playground',                                  92, 255, (80, 80, 80), False, False),
        datasetClass('bench',                                       93, 255, (0, 130, 150), False, False),
        datasetClass('safety_vest_03_orange',                       94, 255, (0, 84, 103), True, False),
        datasetClass('plumbing',                                    95, 255, (190, 150, 230), False, False),
        datasetClass('pergola_garden',                              96, 255, (150, 200, 50), False, False),
        datasetClass('vegetation_road',                             97, 0, (0, 255, 254), False, True),
        datasetClass('stairs',                                      98, 255, (90, 10, 90), False, False),
        datasetClass('safety_vest_02',                              99, 255, (0, 80, 102), True, False),
        datasetClass('sunshades',                                   100, 255, (255, 50, 0), False, False),
        datasetClass('van',                                         101, 14, (185, 255, 75), True, False),
        datasetClass('railings',                                    102, 255, (100, 50, 0), False, False),
        datasetClass('kickbike',                                    103, 255, (185, 255, 46), True, False),
        datasetClass('bin',                                         104, 255, (0, 50, 0), False, False),
        datasetClass('scooter_child',                               105, 255, (225, 35, 25), False, False),
        datasetClass('construction_stock',                          106, 255, (25, 85, 155), False, False),
        datasetClass('traffic_cameras',                             107, 255, (50, 150, 255), False, False),
        datasetClass('construction_pallet',                         108, 255, (111, 131, 111), False, False),
        datasetClass('road',                                        109, 0, (150, 150, 200), False, False),
        datasetClass('electric_post_insulator',                     110, 255, (100, 0, 100), False, False),
        datasetClass('traffic_signs_back',                          111, 255, (0, 200, 0), False, False),
        datasetClass('decoration_garden',                           112, 255, (250, 100, 130), False, False),
        datasetClass('garbage',                                     113, 255, (50, 50, 50), False, False),
        datasetClass('traffic_lights_bulb_green',                   114, 6, (0, 0, 252), False, False),
        datasetClass('cat_ete',                                     115, 255, (5, 50, 150), False, True),
        datasetClass('marquees',                                    116, 255, (180, 90, 30), False, False),
        datasetClass('asphalt_hole',                                117, 0, (213, 128, 213), False, True),
        datasetClass('alley',                                       118, 255, (10, 80, 15), False, False),
        datasetClass('bridge',                                      119, 255, (30, 90, 210), False, False),
        datasetClass('hangar_airport',                              120, 255, (150, 0, 150), False, False),
        datasetClass('billboard',                                   121, 255, (255, 100, 0), False, False),
        datasetClass('barrel',                                      122, 255, (205, 255, 205), False, False),
        datasetClass('toy',                                         123, 255, (60, 30, 255), False, False),
        datasetClass('subway',                                      124, 255, (200, 225, 90), False, False),
        datasetClass('plane',                                       125, 255, (35, 65, 50), False, False),
        datasetClass('food_machine',                                126, 255, (255, 128, 0), False, False),
        datasetClass('runway',                                      127, 255, (50, 50, 200), False, False),
        datasetClass('fire',                                        128, 255, (255, 0, 1), False, False),
        datasetClass('phone_booth',                                 129, 255, (30, 30, 180), False, False),
        datasetClass('skateboard',                                  130, 255, (185, 200, 185), True, False),
        datasetClass('box',                                         131, 255, (110, 90, 225), False, False),
        datasetClass('polished_aggregated',                         132, 0, (213, 0, 0), True, True),
        datasetClass('fire_extinguisher',                           133, 255, (100, 80, 100), False, False),
        datasetClass('people',                                      134, 11, (250, 150, 255), True, False),
        datasetClass('traffic_lights_bulb_red',                     135, 6, (0, 0, 254), False, False),
        datasetClass('sewer_road',                                  136, 0, (255, 0, 1), False, True),
        datasetClass('traffic_lights_poles',                        137, 5, (0, 255, 0), False, False),
        datasetClass('poles',                                       137, 5, (130, 87, 52), False, False),
        datasetClass('polesv2',                                      137, 5, (83, 242, 188), False, False),
        datasetClass('fire_hydrant',                                138, 255, (255, 255, 0), False, False),
        datasetClass('terrain',                                     139, 9, (150, 100, 0), False, False),
        datasetClass('tools',                                       140, 255, (160, 25, 230), False, False),
        datasetClass('safety_vest_01',                              141, 255, (0, 80, 101), True, False),
        datasetClass('safety_vest_03_blue',                         142, 255, (0, 82, 103), True, False),
        datasetClass('sewer',                                       143, 0, (255, 0, 0), False, False),
        datasetClass('stony_floor',                                 144, 255, (150, 100, 100), False, False),
        datasetClass('electric_post',                               145, 255, (100, 0, 200), False, False),
        datasetClass('construction_helmet_yellow',                  146, 255, (0, 80, 200), True, False),
        datasetClass('carpet',                                      147, 255, (230, 30, 130), False, False),
        datasetClass('sky',                                         148, 10, (0, 0, 0), False, False),
        datasetClass('trash_can',                                   149, 20, (255, 150, 150), False, False), ### NEW OOD
        datasetClass('pivot',                                       150, 255, (135, 75, 180), False, False),
        datasetClass('fences',                                      151, 4, (150, 200, 250), False, False),
        datasetClass('construction_helment_orange',                 152, 255, (0, 80, 202), True, False),
        datasetClass('street_lights',                               153, 5, (0, 175, 100), False, False),
        datasetClass('bird',                                        154, 255, (255, 175, 110), False, False),
        datasetClass('Stand food',                                  155, 19, (50, 100, 50), False, False), ### NEW OOD
        datasetClass('Moose',                                       156, 19, (200, 50, 200), False, False), ### NEW OOD
        datasetClass('Deer',                                       157, 19, (50, 100, 144), False, False), ### NEW OOD
        datasetClass('Bear',                                       158, 19, (200, 50, 100), False, False), ### NEW OOD
        datasetClass('Cow',                                       159, 19, (100, 50, 100), False, False), ### NEW OOD
    ]

    train_id_to_color = [c.object_id for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)

    def __init__(self, gt_dataset, pred_dataset, root_odgt, transform=None):
        self.gt_dataset = gt_dataset
        self.pred_dataset = pred_dataset
        self.transform = transform
        self.list_sample = [json.loads(x.rstrip()) for x in open(root_odgt, 'r')]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, segm_path) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        this_record = self.list_sample[index]

        # load predictions, here by default, the outputs are saved as type .pth
        path_out_model = os.path.join(self.pred_dataset, this_record['fpath_img'][:-3] +'pth')
        if not os.path.exists(path_out_model):
            message = "Expected submission file '{0}', found files {1}"
            sys.exit(message.format(path_out_model, os.listdir(self.pred_dataset)))    
        out = torch.load(path_out_model)
        
        # load groundtruth
        segm_path = os.path.join(self.gt_dataset, this_record['fpath_segm'])
        segm = plt.imread(segm_path) * 255.
        target = np.zeros((segm.shape[0], segm.shape[1])) + 255

        for c in self.classes:
            upper = np.array(c.object_id)
            lower = upper
            mask = cv.inRange(segm, lower, upper)
            target[mask == 255] = c.train_id

        target = target.astype(np.uint8)
        target = Image.fromarray(target)
        target = torch.from_numpy( np.array( target, dtype=np.uint8) )
    
        return out, target, segm_path 


    def __len__(self):
        return len(self.list_sample)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data