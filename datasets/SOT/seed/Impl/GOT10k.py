import os
from datasets.types.data_split import DataSplit
from datasets.SOT.constructor.base_interface import SingleObjectTrackingDatasetConstructor
from miscellanies.parser.ini import parse_ini_file
from miscellanies.parser.txt import load_numpy_array_from_txt
from miscellanies.numpy.dtype import try_get_int_array
import ast
import numpy as np


def _construct_GOT10k_public_data(constructor: SingleObjectTrackingDatasetConstructor, sequence_list):
    for sequence_name, sequence_path in sequence_list:
        images = os.listdir(sequence_path)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()
        # print(sequence_name, sequence_path)
        bounding_boxes = load_numpy_array_from_txt(os.path.join(sequence_path, 'groundtruth_rect.txt'), delimiter='\t')
        bounding_boxes = try_get_int_array(bounding_boxes)

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            for image, bounding_box in zip(images, bounding_boxes):
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_path, image))
                    frame_constructor.set_bounding_box(bounding_box.tolist())


def _construct_GOT10k_non_public_data(constructor: SingleObjectTrackingDatasetConstructor, sequence_list):
    for sequence_name, sequence_path in sequence_list:
        images = os.listdir(sequence_path)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        # print("Sequence Name : ", sequence_name)

        bounding_box = load_numpy_array_from_txt(os.path.join(sequence_path, 'groundtruth_rect.txt'), delimiter='\t')
        bounding_box = try_get_int_array(bounding_box)

        # print("BBOX Shape : ", bounding_box.shape)
        # print("BBOX nDIM  : ", bounding_box.ndim)
        assert bounding_box.ndim == 1 and bounding_box.shape[0] == 4

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            for index_of_image, image in enumerate(images):
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_path, image))
                    if index_of_image == 0:
                        frame_constructor.set_bounding_box(bounding_box.tolist())


def construct_GOT10k(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split
    sequence_filter = seed.sequence_filter

    if data_split == DataSplit.Training:
        folder = 'train'
    elif data_split == DataSplit.Validation:
        folder = 'val'
    elif data_split == DataSplit.Testing:
        folder = 'test'
    else:
        raise RuntimeError(f'Unsupported dataset split {data_split}')

    # constructor.set_category_id_name_map({k: v for k, v in enumerate(_category_names)})
    print("Loading False HSI")
    sequence_list = []
    for sequence_name in os.listdir(os.path.join(root_path, folder)):
        sequence_name = sequence_name.strip()
        current_sequence_path = os.path.join(root_path, folder, sequence_name) + '/HSI-FalseColor'
        # print("path : ", current_sequence_path)
        sequence_list.append((sequence_name, current_sequence_path))

    constructor.set_total_number_of_sequences(len(sequence_list))

    _construct_GOT10k_public_data(constructor, sequence_list)
