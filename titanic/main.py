import pandas as pd

def read_data(path="./data/"):
  return pd.read_csv(path,skip_blank_lines=True)

TRAIN_PATH = "./data/train/train.csv"
TEST_PATH = "./data/test/test.csv"
if __name__ == '__main__':
  train_data = read_data(TRAIN_PATH)
  print(train_data)



