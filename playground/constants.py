import operator

IMAGE_SIZE = 32
PIX_FROM_MID = IMAGE_SIZE / 2
NUM_CLASSES = 6

CHECKPOINT_DIR="../model/my_model/"

CLASS_NAMES = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups", "unknown"]

### Encoded labels taken from train_test_prep.encode_label()
ENCODED_NAMES = [1, 2, 5, 3, 0, 4]

ENCODED_DICT = {"adult_males":1, "subadult_males":2, "adult_females":5, "juveniles":3, "pups":0, "unknown":4}
DECODED_DICT = {1:"adult_males", 2:"subadult_males", 5:"adult_females", 3:"juveniles", 0:"pups", 4:"unknown"}

ENCODED_SORTED = sorted(ENCODED_DICT.items(), key=operator.itemgetter(1))
CLASS_NAMES_ENC_SORTED = [x[0] for x in ENCODED_SORTED]
