class AbstractAgent:

    def __init__(self):
        pass

    def play_round(self, player_idx, round_num, recieved, popularities, influence, extra_data, v, transactions):
        pass

    def setGameParams(self, gameParams, visualTraits, _forcedRandom):
        pass

    def record_final_results(self, player_idx, round_num, recieved, popularities, influence, extra_data, v,
                             transactions):
        pass
