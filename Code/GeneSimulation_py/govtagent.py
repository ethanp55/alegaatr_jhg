from baseagent import AbstractAgent

import numpy as np

class DummyGovtAgent(AbstractAgent):

    def __init__(self, tax_rate):
        super().__init__()
        self.whoami = "Govment"
        self.tax_rate = tax_rate
        self.allow_keep = False

    def get_player_taxes(self, player_idx, recieved, popularities, influence, extra_data):
        tkns = len(popularities) * 2
        taxes = self.tax_rate * tkns
        return taxes

    def play_round(self, player_idx, round_num, recieved, popularities, influence, extra_data):
        allocations = np.zeros(len(popularities), dtype=int)
        allocations[player_idx] = len(popularities) * 2
        return allocations

    def setGameParams(self, gameParams, visualTraits, _forcedRandom):
        self.gameParams = gameParams
