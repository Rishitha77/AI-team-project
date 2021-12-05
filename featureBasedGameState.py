# from model import Model
# from pacman import GameState
# The 2 imports above ^ are not required for the code but IDEs can use them for typing hints. But comment it again before
# running anything though - they break a few things

import search
import searchAgents


class FeatureBasedGameState(object):
    def __init__(self, gameState):
        if gameState == None:
            return
        self.rawGameState = gameState
        self.moveToClosestFood = None
        self.ghostWithin1UnitOfClosestFoodDirectionPoint = None
        self.closestGhosts = None
        self.ghostPositions = self.rawGameState.getGhostPositions()
        self.moveToClosestFood = self.getMoveToClosestFood()
        self.ghostWithin1UnitOfClosestFoodDirectionPoint = self.isGhostWithin1UnitOfClosestFoodDirectionPoint()

    def getMoveToClosestFood(self):
        return search.aStarSearch(searchAgents.AnyFoodSearchProblem(self.rawGameState))[0]

    def isGhostWithin1UnitOfClosestFoodDirectionPoint(self):
        x, y = self.rawGameState.getPacmanPosition()
        closestFoodMovePoint = None
        if self.moveToClosestFood == "North":
            closestFoodMovePoint = (x, y + 1)
        elif self.moveToClosestFood == "South":
            closestFoodMovePoint = (x, y - 1)
        elif self.moveToClosestFood == "East":
            closestFoodMovePoint = (x + 1, y)
        elif self.moveToClosestFood == "West":
            closestFoodMovePoint = (x - 1, y)
        else:
            raise Exception("Invalid move " + str(self.moveToClosestFood))
        cpx, cpy = closestFoodMovePoint
        intersection = {(cpx, cpy), (cpx + 1, cpy), (cpx - 1, cpy), (cpx, cpy + 1), (cpx, cpy - 1)} & set(self.ghostPositions)
        return len(intersection) > 0

    def __key(self):
        return (
            self.moveToClosestFood,
            self.ghostWithin1UnitOfClosestFoodDirectionPoint
        )

    # If you add a feature, add it here too
    # This is just to print the features properly in model.txt
    def __repr__(self):
        return str({
            "moveToClosestFood": self.moveToClosestFood,
            "ghostWithin1UnitOfClosestFoodDirectionPoint": self.ghostWithin1UnitOfClosestFoodDirectionPoint
        })

    # If you add a feature, add it here too
    # This is required by pickle to store the model from the file
    def __getstate__(self):
        return (self.moveToClosestFood, self.ghostWithin1UnitOfClosestFoodDirectionPoint)

    # If you add a feature, add it here too
    # This is required by pickle to retrieve the model from the file
    def __setstate__(self, state):
        self.moveToClosestFood = state[0]
        self.ghostWithin1UnitOfClosestFoodDirectionPoint = state[1]

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, FeatureBasedGameState):
            return self.__key() == other.__key()
        return NotImplemented
