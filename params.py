class ModelParams:
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

class LossParams:
    ALPHA = 0.5
    BETA = 0.5
    GAMMA = 2
    THRESHOLD = 0.5

class TrainParams:
    EPOCHS = 20
    BATCH_SIZE = 16
    PLOT_EVERY = 100
    NUM_WORKERS = 4
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
    UNSEEN_CLASSES = [
        
    ]