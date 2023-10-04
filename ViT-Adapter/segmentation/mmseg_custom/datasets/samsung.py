from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class samsung(CustomDataset):

        CLASSES=('Road', 'Sidewalk', 'Construction', 'Fence', 'Pole', 'Traffic Light',
                 'Traffic Sign', 'Nature', 'Sky', 'Person', 'Rider', 'Car','Background')
        PALETTE=[[128, 64, 128], [244, 35, 232], [0, 0, 0], [190, 153, 153],
                 [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35],
                  [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [255,0,120]]
        label_map={0: 0,  # Road
                   1: 1,  # Sidewalk
                   2: 2,  # Construction
                   3: 3,  # Fence
                   4: 4,  # Pole
                   5: 5,  # Traffic Light
                   6: 6,  # Traffic Sign
                   7: 7,  # Nature
                   8: 8,  # Sky
                   9: 9,  # Person
                   10: 10,  # Rider
                   11: 11, # Car
                  12: 12,}  #Background
        def __init__(self, **kwargs):
            super(samsung, self).__init__(
                img_suffix='.png',
                seg_map_suffix='.png',
                reduce_zero_label=False,
                **kwargs)

        