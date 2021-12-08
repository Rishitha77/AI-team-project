from collections import namedtuple
from featureBasedGameState import FeatureBasedGameState
import pickle
import constants

ModelEntry = namedtuple('ModelEntry', "nWins nSimulations avgReward pseudoWins")

class Model(object):
    def __init__(self):
        self.data = {}

    def updateState(self, gamestate, nextAction, wins, pwins, simulations, reward):
        self.data[(gamestate, nextAction)] = ModelEntry(nWins=wins, pseudoWins=pwins,
                                                    nSimulations=simulations, avgReward=reward)

    def newModel(self, file):
        with open(file, 'w') as f:
            for k, v in list(self.data.items()):
                f.write(str(k) + ": " + str(v) + "\n")
        self.savetoFile(constants.OUTPUT_MODEL)

    def savetoFile(self, filePath):
        filename = filePath
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)


def loadModel(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    model = Model()
    model.data = data
    return model

commonModel = None
if constants.MODEL is not None:
    commonModel = loadModel(constants.MODEL)
else:
    commonModel = Model()
