from enum import Enum
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Scaler(Enum):
    MINMAX = MinMaxScaler()
    STANDARD = StandardScaler()

# Conservative trading fees according to Binanace
class Trading_conditions(Enum):
    MAKER_FEE = 0.001
    TAKER_FEE = 0.001

