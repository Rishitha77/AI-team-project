
import search
import searchAgents

class FeatureBasedGameState(object):
    def __init__(self, gameState):
        if gameState == None:
            return
        self.oldGameState = gameState
        self.ghostNearFood = None
        self.closestFood = None
        self.closestGhosts = None
        self.ghostPositions = self.oldGameState.getGhostPositions()
        self.closestFood = self.getMoveToClosestFood()
        self.ghostNearFood = self.isghostNearFood()

    def getMoveToClosestFood(self):
        return search.aStarSearch(searchAgents.AnyFoodSearchProblem(self.oldGameState))[0]

    def isghostNearFood(self):
        x, y = self.oldGameState.getPacmanPosition()
        moveNearFood = None
        if self.closestFood == "North":
            moveNearFood = (x, y + 1)
        elif self.closestFood == "East":
            moveNearFood = (x + 1, y)
        elif self.closestFood == "West":
            moveNearFood = (x - 1, y)
        elif self.closestFood == "South":
            moveNearFood = (x, y - 1)
        else:
            raise Exception("Not a valid move " + str(self.closestFood))
        cpx, cpy = moveNearFood
        intersection = {(cpx, cpy), (cpx + 1, cpy), (cpx - 1, cpy), (cpx, cpy + 1), (cpx, cpy - 1)} & set(self.ghostPositions)
        return len(intersection) > 0

    def __key(self):
        return (
            self.closestFood,
            self.ghostNearFood
        )

    # To print the features properly in model.txt
    def __repr__(self):
        return str({
            "closestFood": self.closestFood,
            "ghostNearFood": self.ghostNearFood
        })

    # To store the model from the file
    def __getstate__(self):
        return (self.closestFood, self.ghostNearFood)

    # To retrieve the model from the file
    def __setstate__(self, state):
        self.closestFood = state[0]
        self.ghostNearFood = state[1]

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, FeatureBasedGameState):
            return self.__key() == other.__key()
        return NotImplemented
