import enum

TRIAL = False
SANITY_CHECK = False
VERBOSE = True
CONV_LAYER_FROZEN = False
BATCH_SIZE = 20
VALIDATION_STEPS = 50
PREDICTION_STEPS = 25
INPUT_SHAPE = (384, 384, 3)
SAVE_MODEL_WEIGHTS = True

if TRIAL:
    STEPS_PER_EPOCH = 2
    EPOCHS = 2
    STEPS_PER_EPOCH_MODEL_2 = 2
    EPOCHS_MODEL_2 = 2
    seeds = [1970, 1972]
else:
    STEPS_PER_EPOCH = 100
    EPOCHS = 30
    STEPS_PER_EPOCH_MODEL_2 = 40
    EPOCHS_MODEL_2 = 60
    seeds = [1970, 1972, 2008, 2019, 2020]


# ---------
# ENUMERATIONS
# ---------
class HINTS_TYPE(enum.Enum):
    ASYMMETRY = 1
    BORDER = 2
    COLOR = 3


class HINTS_SOURCE(enum.Enum):
    STUDENT = 1
    MTURK = 2  # m-turk batch files June 2020
    AUTOMATED = 3
    EVERYONE = 4


class NETWORK_TYPE(enum.Enum):
    VGG16 = 1
    INCEPTION = 2
    RESNET = 3


class ENSEMBLE_LEARNING_TYPE(enum.Enum):
    SOFT_VOTING = 1
    AVERAGING = 2
    OPTIMIZING = 3


NETWORK_SELECTED = NETWORK_TYPE.VGG16
ENSEMBLE_LEARNING = ENSEMBLE_LEARNING_TYPE.OPTIMIZING
SOURCE_SELECTED = HINTS_SOURCE.AUTOMATED


def print_constants():
    print('*************************************************************')
    print('SANITY_CHECK           = ', str(SANITY_CHECK))
    print('CONV_LAYER_FROZEN      = ', str(CONV_LAYER_FROZEN))
    print('SAVE_MODEL_WEIGHTS     = ', str(SAVE_MODEL_WEIGHTS))
    print('SELECTED_NETWORK       = ', str(NETWORK_SELECTED))
    print('SOURCE_SELECTED        = ', str(SOURCE_SELECTED))
    print('ENSEMBLE_LEARNING      = ', str(ENSEMBLE_LEARNING))
    print('TRIAL                  = ', str(TRIAL))
    print('*************************************************************')
