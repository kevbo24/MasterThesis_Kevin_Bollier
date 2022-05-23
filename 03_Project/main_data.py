from src.data_prep import Preprocessing, Train_Test_Split
from src.data_enums import Scaler

def main():
    # prepare data
    prep = Preprocessing(Scaler.MINMAX)
    prep.prep_data()

    # split data into train and test
    train_test = Train_Test_Split()
    train_test.train_test_split()


if __name__ == '__main__':
    main()


