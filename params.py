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
    ALPHA = 1
    BETA = 1
    GAMMA = 2
    THRESHOLD = 0.5

class TrainParams:
    EPOCHS = 20
    BATCH_SIZE = 16
    PLOT_EVERY = 100