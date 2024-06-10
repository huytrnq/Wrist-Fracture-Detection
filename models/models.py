from joblib import load
from joblib import dump

import os
import numpy as np
import skimage
import pandas as pd

from utils.dataset import sliding_window
from utils.bboxes import iou, merge_two_boxes, get_original_coordinates


def export_model(model, filename):
    """
    Exports a trained scikit-learn model to a file using joblib.

    Args:
    - model (sklearn.base.BaseEstimator): The trained model to export.
    - filename (str): The path to the file where the model will be saved.
    """
    dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Loads a scikit-learn model from a file using joblib.

    Args:
    - filename (str): The path to the file from which the model will be loaded.

    Returns:
    - model (sklearn.base.BaseEstimator): The loaded scikit-learn model.
    """
    model = load(filename)
    print(f"Model loaded from {filename}")
    return model


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "weights/fast_region_proposal.pkl"
)
DEFAULT_SCALER_PATH = os.path.join(os.path.dirname(__file__), "weights/scaler.pkl")


class WristFractureDetection:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        step_size=64,
        window_size=64,
        pool_size=(4, 4),
        feature_extractor=None,
        scaler=DEFAULT_SCALER_PATH,
    ):
        """Wrist Fracture Detection

        Args:
            model_path (str, optional): Path to exported model. Defaults to None.
            step_size (int, optional): Stride step. Defaults to 128.
            window_size (int, optional): Sliding window size. Defaults to 256.
            pool_size (tuple, optional): Pooling window size. Defaults to (4, 4).
            feature_extractor (function, optional): Feature extractor. Defaults to None.
            scaler (str, optional): Path to scaler. Defaults to './scaler.pkl'.
        """
        self.model = self.load(model_path) if model_path else None
        self.step_size = step_size
        self.window_size = window_size
        self.pool_size = pool_size
        self.feature_extractor = feature_extractor
        self.scaler = load(scaler)
        self.feature_seletion = FeatureSelection()

    def padding(self, image, factor=64, mode="constant", value=0):
        """
        Pads an image so that its dimensions are multiples of a given factor.

        Args:
            image (np.ndarray): The input image.
            factor (int): The factor to which the dimensions should be multiples of.
            mode (str): The mode parameter for np.pad.
            value (int): The padding value to use when mode is 'constant'.

        Returns:
            np.ndarray: The padded image.
        """
        # Get the dimensions of the image
        h, w = image.shape[:2]

        # Calculate the target dimensions
        target_h = ((h + factor - 1) // factor) * factor
        target_w = ((w + factor - 1) // factor) * factor

        # Calculate the padding amounts
        pad_h = target_h - h
        pad_w = target_w - w

        # Pad the image
        if len(image.shape) == 3:  # Color image
            padded_image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode=mode,
                constant_values=value,
            )
        else:  # Grayscale image
            padded_image = np.pad(
                image, ((0, pad_h), (0, pad_w)), mode=mode, constant_values=value
            )

        return padded_image

    def predict(self, image, offset):
        """Predict

        Args:
            X (np.array): Features

        Returns:
            Predictions: np.array
        """
        rois = []
        scaled_roi = []

        feature_windows = []
        image = self.padding(image, factor=self.window_size)
        for x, y, window in sliding_window(image, self.step_size, self.window_size):
            if (
                window.shape[0] != self.window_size
                or window.shape[1] != self.window_size
            ):
                continue
            ## If zero percentage is more than 50% then skip
            zero_percentage = np.mean(window == 0) * 100
            if zero_percentage > 50:
                continue
            window = skimage.measure.block_reduce(window, (2, 2), np.max)

            features = self.feature_extraction(window)
            features = self.scaler.transform(features.reshape(1, -1)).ravel()
            feature_windows.append(features)
            rois.append((x, y, x + self.window_size, y + self.window_size))

        if len(feature_windows) > 0:
            feature_windows = self.feature_seletion(feature_windows)
            results = self.model.predict(feature_windows)
        else:
            results = []
            return results

        ### Get the rois which contain the fracture
        rois = np.array(rois)
        selected_rois = rois[results == 1]
        for roi in selected_rois:
            roi = get_original_coordinates(roi, self.pool_size)
            ## Add the offset to the roi
            roi = (
                roi[0] + offset[0],
                roi[1] + offset[1],
                roi[2] + offset[0],
                roi[3] + offset[1],
            )
            scaled_roi.append(roi)
        # return self.merge_boxes(scaled_roi)
        return scaled_roi

    def load(self, filename):
        """Load the model

        Args:
            filename (str): Path to load the model
        """
        return load_model(filename)

    def feature_extraction(self, img):
        """Feature extraction

        Args:
            img (np.array): Image

        Returns:
            np.array: Features
        """
        if type(self.feature_extractor) is not list:
            return self.feature_extractor(img).ravel()
        else:
            features = []
            for extractor in self.feature_extractor:
                features.append(extractor(img).ravel())
            return np.concatenate(features)

    def merge_boxes(self, boxes, iou_threshold=0.2):
        """
        Merge overlapping bounding boxes.

        Args:
            boxes (numpy.ndarray): Array of bounding boxes of shape (N, 4), where N is the number of boxes.
                                Each box is represented as [x1, y1, x2, y2].
            iou_threshold (float): Threshold for IoU to merge overlapping boxes.

        Returns:
            numpy.ndarray: Array of merged bounding boxes.
        """
        if len(boxes) == 0:
            return np.array([])
        boxes = np.array(boxes)

        # Convert boxes to float if they are not already
        if boxes.dtype.kind != "f":
            boxes = boxes.astype(np.float32)

        merged_boxes = []
        used = np.zeros(len(boxes), dtype=bool)

        for i in range(len(boxes)):
            if used[i]:
                continue
            current_box = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if iou(current_box, boxes[j]) >= iou_threshold:
                    current_box = merge_two_boxes(current_box, boxes[j])
                    used[j] = True
            merged_boxes.append(current_box)
            used[i] = True

        return np.array(merged_boxes).astype(int).tolist()


mRMR_features = [
    80,
    661,
    606,
    558,
    71,
    431,
    494,
    141,
    25,
    577,
    64,
    353,
    134,
    238,
    173,
    72,
    302,
    143,
    31,
    318,
    23,
    16,
    365,
    128,
    73,
    545,
    598,
    21,
    79,
    446,
    604,
    95,
    125,
    24,
    398,
    86,
    109,
    701,
    430,
    366,
    22,
    305,
    583,
    20,
    317,
    69,
    462,
    136,
    70,
    98,
    497,
    487,
    333,
    135,
    584,
    137,
    237,
    254,
    68,
    124,
    773,
    239,
    93,
    88,
    590,
    439,
    74,
    591,
    526,
    172,
    138,
    597,
    133,
    50,
    61,
    87,
    45,
    525,
    859,
    67,
    89,
    513,
    611,
    26,
    7,
    674,
    108,
    478,
    301,
    85,
    596,
    746,
    5,
    285,
    114,
    618,
    636,
    503,
    269,
    6,
    424,
    397,
    605,
    75,
    703,
    399,
    107,
    665,
    477,
    90,
    841,
    104,
    19,
    809,
    46,
    502,
    683,
    414,
    465,
    139,
    205,
    290,
    440,
    429,
    332,
    561,
    84,
    638,
    420,
    162,
    501,
    670,
    423,
    663,
    253,
    59,
    140,
    488,
    589,
    510,
    132,
    9,
    928,
    656,
    634,
    416,
    559,
    76,
    242,
    56,
    284,
    551,
    576,
    27,
    60,
    169,
    221,
    177,
    264,
    495,
    719,
    206,
    481,
    268,
    549,
    612,
    274,
    504,
    102,
    667,
    267,
    126,
    171,
    210,
    940,
    91,
    48,
    150,
    99,
    92,
    838,
    17,
    4,
    174,
    594,
    469,
    461,
    54,
    676,
    917,
    121,
    208,
    322,
    647,
    550,
    601,
    216,
    753,
    782,
    219,
    178,
    384,
    258,
    44,
    329,
    737,
    123,
    567,
    699,
    506,
    105,
    613,
    413,
    153,
    18,
    354,
    603,
    77,
    57,
    422,
    843,
    471,
    433,
    149,
    66,
    96,
    106,
    281,
    608,
    751,
    51,
    827,
    220,
    34,
    331,
    546,
    214,
    599,
    334,
    580,
    262,
    289,
    535,
    28,
    493,
    587,
    511,
    672,
    224,
    620,
    890,
    58,
    489,
    582,
    256,
    426,
    517,
    204,
    744,
    369,
    445,
    421,
    554,
    306,
    8,
    706,
    100,
    65,
    552,
    624,
    338,
    217,
    470,
    697,
    811,
    3,
    739,
    681,
    13,
    441,
    560,
    402,
    11,
    472,
    721,
    649,
    283,
    265,
    166,
    775,
    408,
    771,
    505,
    146,
    640,
    55,
    389,
    163,
    63,
    154,
    286,
    103,
    575,
    685,
    218,
    211,
    931,
    627,
    615,
    485,
    784,
    309,
    450,
    391,
    447,
    113,
    223,
    266,
    610,
    170,
    115,
    816,
    692,
    259,
    519,
    118,
    241,
    110,
    800,
    622,
    553,
    592,
    456,
    185,
    895,
    337,
    101,
    619,
    326,
    514,
    518,
    617,
    654,
    111,
    520,
    631,
    52,
    33,
    357,
    225,
    694,
    271,
    498,
    847,
    532,
    425,
    122,
    112,
    757,
    726,
    29,
    15,
    215,
    588,
    865,
    311,
    375,
    845,
    345,
    500,
    437,
    581,
    263,
    544,
    10,
    534,
    436,
    629,
    278,
    2,
    53,
    574,
    260,
    35,
    168,
    272,
    735,
    438,
    417,
    47,
    595,
    330,
    484,
    165,
    486,
    527,
    401,
    614,
    212,
    359,
    207,
    120,
    248,
    368,
    236,
    152,
    881,
    388,
    621,
    507,
    733,
    231,
    454,
    273,
    870,
    186,
    40,
    312,
    127,
    449,
    30,
    117,
    229,
    161,
    913,
    323,
    406,
    528,
    645,
    496,
    442,
    407,
    83,
    287,
    261,
    868,
    899,
    151,
    374,
    194,
    490,
    310,
    160,
    270,
    370,
    82,
    360,
    346,
    818,
    557,
    282,
    300,
    807,
    453,
    175,
    548,
    39,
    555,
    748,
    358,
    791,
    586,
    328,
    213,
    602,
    367,
    404,
    474,
    275,
    755,
    335,
    712,
    455,
    607,
    129,
    43,
    296,
    387,
    372,
    320,
    36,
    184,
    658,
    861,
    14,
    405,
    904,
    167,
    479,
    38,
    193,
    116,
    708,
    565,
    679,
    466,
    280,
    585,
    144,
    12,
    164,
    0,
    579,
    593,
    252,
    392,
    190,
    203,
    452,
    199,
    325,
    571,
    131,
    303,
    572,
    182,
    728,
    390,
    463,
    81,
    385,
    198,
    566,
    119,
    652,
    529,
    468,
    97,
    195,
    327,
    200,
    130,
    569,
    459,
    176,
    183,
    600,
    820,
    246,
    427,
    509,
    342,
    625,
    316,
    364,
    688,
    383,
    473,
    62,
    730,
    780,
    247,
    41,
    522,
    382,
    319,
    396,
    159,
    457,
    277,
    343,
    564,
    717,
    886,
    411,
    393,
    643,
    336,
    512,
    291,
    834,
    279,
    196,
    616,
    321,
    491,
    294,
    573,
    228,
    854,
    823,
    295,
    609,
    475,
    924,
    179,
    562,
    192,
    443,
    157,
    458,
    710,
    395,
    201,
    521,
    344,
    409,
    901,
    568,
    244,
]


class FeatureSelection:
    """
    Select the features that are selected in the constructor
    """
    def __init__(self, feature_indexes=mRMR_features):
        self.selected_features = feature_indexes
        

    def __call__(self, features):
        """Select the features that are selected in the constructor

        Args:
            features (np.array): The features to select
        """
        features_pd = pd.DataFrame(features)
        features = features_pd[self.selected_features].values
        return features
