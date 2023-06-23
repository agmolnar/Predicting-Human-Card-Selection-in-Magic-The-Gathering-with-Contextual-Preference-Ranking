import pickle
import numpy as np
df = np.load(r"picks\0.npy", allow_pickle=True)
type(df)
df[0]

df = pickle.load(open(r"test_data/test_data0.pt", "rb"))
df = pickle.load(open(r"val_data/val_data0.pt", "rb"))
df = pickle.load(open(r"training_data/train_data0.pt", "rb"))
df[8]
df = pickle.load(open(r"training_datasets/train_data0.pt", "rb"))
df.positives[0]
df.negatives[0]
df.anchors[0]
type(df)
len(df)
len(df.negatives)
df.positives[997925]
df.negatives[997925]
sum(sum(df.negatives))
sum(sum(df.positives))
sum(sum(df.anchors))

df = pickle.load(open(r"Data/card_dictDMU.pt", "rb"))
df[10001]

# 273 training examples per draft (14 cards per pack)
# 177,299 drafts -> 48,402,627 examples total

# RANDOM BOT: 23.22% mean accuracy

# only a single epoch of 50 of those datasets was used

# load very small subset of csv
# need to create nbr of cards in pack x 3 torch.Tensor for each pick (line)
# ask how to make model train directly on data instead of creating file for it, for efficiency/storage issues
# 0. Create new cols: cards_in_pack, anchors, positives, negatives
# 1. anchors, already created (pool_card), repeated over the number of cards in pack
#    just need to create new col and create one-hot tensor of it
# 2. positives, already created (pick), similar as for anchors
# 3. negatives, representation of each possible WRONG pick
# get inspired from their code
# if cards_in_pack = 1, then you del line and move on
# ultimately, each pick/line should result into anywhere between 2 and 13 lines (training examples)
