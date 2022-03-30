class Reward:
    def __init__(self):
        pass
    def calculateReward(self):
        return 0

def getRewardDict(rewards):
    rewardDict = {}
    for reward in rewards:
        rewardDict[reward.__class__.__name__] = 0
    return rewardDict