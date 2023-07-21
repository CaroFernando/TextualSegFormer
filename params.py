class DEPRECATED_ModelParams:
    INCHANNELS = 3
    WIDTHS = [64, 128, 256, 512]
    DEPTHS = [3, 3, 3, 3]
    NUM_HEADS = [4, 8, 16, 32]
    PATCH_SIZES = [7, 5, 3, 3]
    OVERLAP_SIZES = [4, 2, 2, 2]
    REDUCTION_RATIOS = [8, 4, 2, 1]
    EXPANSION_FACTORS = [2, 2, 2, 2]
    DECODER_CHANNELS = 512
    SCALE_FACTORS = [8, 4, 2, 1]

class ModelParams:
    INCHANNELS = 3
    WIDTHS = [64, 128, 256, 512]
    DEPTHS = [2, 2, 2, 2]
    NUM_HEADS = [4, 8, 16, 32]
    PATCH_SIZES = [7, 5, 3, 3]
    OVERLAP_SIZES = [4, 2, 2, 2]
    REDUCTION_RATIOS = [8, 4, 2, 1]
    EXPANSION_FACTORS = [2, 2, 2, 2]
    DECODER_CHANNELS = 512
    SCALE_FACTORS = [8, 4, 2, 1] 

class OptimParams:
    LR = 0.2
    WARMUP_STEPS = 1000
    BETAS = (0.9, 0.99)

class LossParams:
    ALPHA = 0.5
    BETA = 0.5
    GAMMA = 2
    THRESHOLD = 0.5

class TrainParams:
    DATASET_IMAGE_FOLDER = 'ProcessedDatasetStuff512/images/'
    DATASET_IMAGE_FOLDER_TRAIN = 'ProcessedDatasetStuff512/images/train/'
    DATASET_MASK_FOLDER_TRAIN = 'ProcessedDatasetStuff512/masks/train/'
    DATASET_IMAGE_FOLDER_VAL = 'ProcessedDatasetStuff512/images/val/'
    DATASET_MASK_FOLDER_VAL = 'ProcessedDatasetStuff512/masks/val/'
    TRAIN_CSV_PATH = 'ProcessedDatasetStuff512/csv/train.csv'
    IMAGE_DIM = 512
    IMAGE_SIZE = (512, 512)
    MASK_SIZE = 512//4
    EPOCHS = 10
    BATCH_SIZE = 8
    PLOT_EVERY = 100
    NUM_WORKERS = 2
    TRAIN_VAL_SPLIT = 0.8
    TEMPLATES = [
        "A photo of a {}.",
        "A photo of a small {}.",
        "A photo of a medium {}.",
        "A photo of a large {}.",
        "This is a photo of a {}.",
        "This is a photo of a small {}.",
        "This is a photo of a medium {}.",
        "This is a photo of a large {}.",
        "A {} in the scene.",
        "A photo of a {} in the scene.",
        "There is a {} in the scene.",
        "There is the {} in the scene.",
        "This is a {} in the scene.",
        "This is the {} in the scene.",
        "This is one {} in the scene."
    ]
    BOTH_CLASSES = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable']
    SEEN_CLASSES = ['unlabeled',
        'person',
        'bicycle',
        'car',
        'motorcycle',
        'airplane',
        'bus',
        'train',
        'truck',
        'boat',
        'traffic light',
        'fire hydrant',
        'street sign',
        'stop sign',
        'parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'horse',
        'sheep',
        'elephant',
        'bear',
        'zebra',
        'hat',
        'backpack',
        'umbrella',
        'shoe',
        'eye glasses',
        'handbag',
        'tie',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'surfboard',
        'tennis racket',
        'bottle',
        'plate',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'couch',
        'potted plant',
        'bed',
        'mirror',
        'dining table',
        'window',
        'desk',
        'toilet',
        'door',
        'tv',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'blender',
        'book',
        'clock',
        'vase',
        'teddy bear',
        'hair drier',
        'toothbrush',
        'hair brush',
        'banner',
        'blanket',
        'branch',
        'bridge',
        'building-other',
        'bush',
        'cabinet',
        'cage',
        'carpet',
        'ceiling-other',
        'ceiling-tile',
        'cloth',
        'clothes',
        'counter',
        'cupboard',
        'curtain',
        'desk-stuff',
        'dirt',
        'door-stuff',
        'fence',
        'floor-marble',
        'floor-other',
        'floor-stone',
        'floor-tile',
        'floor-wood',
        'flower',
        'fog',
        'food-other',
        'fruit',
        'furniture-other',
        'gravel',
        'ground-other',
        'hill',
        'house',
        'leaves',
        'light',
        'mat',
        'metal',
        'mirror-stuff',
        'moss',
        'mountain',
        'mud',
        'napkin',
        'net',
        'paper',
        'pavement',
        'pillow',
        'plant-other',
        'plastic',
        'platform',
        'railing',
        'railroad',
        'rock',
        'roof',
        'rug',
        'salad',
        'sand',
        'sea',
        'shelf',
        'sky-other',
        'skyscraper',
        'snow',
        'solid-other',
        'stairs',
        'stone',
        'straw',
        'structural-other',
        'table',
        'tent',
        'textile-other',
        'towel',
        'vegetable']
    UNSEEN_CLASSES = ['cow',
        'giraffe',
        'suitcase',
        'frisbee',
        'skateboard',
        'carrot',
        'scissors',
        'cardboard',
        'clouds',
        'grass',
        'playingfield',
        'river',
        'road',
        'tree']
    
    UNSEEN_CLASSES_DEPRECATED = ['horse', 'bear', 'umbrella', 'shoe', 'snowboard', 'bowl', 'keyboard', 
                      'teddy bear', 'branch', 'door-stuff', 'mirror-stuff', 'mud', 'napkin', 
                      'sky-other', 'solid-other']
    SEEN_CLASSES_DEPRECATED = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                    'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 
                    'bird', 'cat', 'dog', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe', 'hat', 'backpack', 'eye glasses', 
                    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 
                    'spoon', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
                    'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 
                    'door', 'tv', 'laptop', 'mouse', 'remote', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
                    'blender', 'book', 'clock', 'vase', 'scissors', 'hair drier', 'toothbrush', 'hair brush', 'banner', 'blanket', 
                    'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 
                    'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'fence', 'floor-marble', 
                    'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 
                    'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'moss', 'mountain', 'net', 
                    'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 
                    'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'skyscraper', 'snow', 'stairs', 'stone', 'straw', 
                    'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable']
    