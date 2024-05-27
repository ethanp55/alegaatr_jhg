from baseagent import AbstractAgent
from os.path import exists

import numpy as np
import os
import time

class AssassinAgent(AbstractAgent):

    def __init__(self):
        super().__init__()
        self.whoami = "Assassin"
        self.gameParams = {}


    def setGameParams(self, gameParams, _forcedRandom):
        self.gameParams = gameParams
        self.force_random = _forcedRandom


    def play_round(self, player_idx, round_num, received, popularities, influence, extra_data):
        num_players = len(popularities)
        num_tokens = num_players * 2
        allocations = np.zeros(len(popularities), dtype=int)

        if round_num == 0:
            self.init_vars(num_players)
            allocations[player_idx] = num_tokens
        else:
            self.update_vars(num_players, player_idx, influence)
            keep_tokens = 0

            if popularities[player_idx] >= 0.5:
                attacked = self.attacks_on_self(num_players, received, popularities)
                w = 0.65
                self.attacks_on_me = w * attacked + (1-w) * self.attacks_on_me
                keep_tokens = min(int((self.attacks_on_me / popularities[player_idx]) + 0.5), num_tokens)

                print(keep_tokens)

                # decide who to attack with remaining tokens
                prey_idx = -1
                prey_pop = 99999.0
                attack_proportion = 0.0
                prop_assassin_attack = self.get_my_proportion(player_idx)
                attack_power = (popularities[player_idx] / prop_assassin_attack) * self.gameParams["steal"] * self.gameParams["alpha"]

                for i in range(0, num_players):
                    if i not in self.the_assassins:
                        print('possible assassin: ' + str(i))
                        if (popularities[i] < prey_pop) and (popularities[i] >= (attack_power / 2.0)):
                            prey_idx = i
                            prey_pop = popularities[i]
                            attack_proportion = min(popularities[i] / attack_power, 1.0)

                print('prey_idx: ' + str(prey_idx))

                # allocate keep and steal
                if prey_idx != -1:
                    steal_tokens = int((attack_proportion * 0.9) * (num_tokens - keep_tokens))
                    allocations[prey_idx] = -steal_tokens
                    allocations[player_idx] = num_tokens - steal_tokens
                else:
                    allocations[player_idx] = num_tokens
            else:
                allocations[player_idx] = num_tokens

        return allocations


    def init_vars(self, num_players):
        self.attacks_by = np.zeros(num_players, dtype=float)
        self.gives_by = np.zeros(num_players, dtype=float)
        self.did_nono = np.full(num_players, False, dtype=bool)
        self.the_assassins = set(range(num_players))
        self.attacks_on_me = 0.0


    def update_vars(self, num_players, player_idx, influence):
        # update the set of assassins
        self.attacks_by = np.zeros(num_players, dtype=float)
        self.gives_by = np.zeros(num_players, dtype=float)
        self.did_nono = np.full(num_players, False, dtype=bool)
        for i in range(0, num_players):
            if i in self.the_assassins:
                for j in range(0, num_players):
                    if i == j:
                        continue

                    if influence[i][j] > 0.0:
                        self.attacks_by[i] -= influence[i][j]
                        if j in self.the_assassins:
                            self.did_nono[i] = True
                    elif influence[i][j] > 0.0:
                        self.gives_by[i] += influence[i][j]
                        self.did_nono[i] = True

        for i in range(0, num_players):
            if (i in self.the_assassins) and (self.did_nono[i] or ((self.attacks_by[player_idx] > 0.0) and (self.attacks_by[i] == 0.0))):
                self.the_assassins.remove(i)

    
    def attacks_on_self(self, num_players, received, popularities):
        amount = 0.0
        for i in range(0, num_players):
            if received[i] < 0.0:
                amount += received[i] * popularities[i]

        return -amount


    def get_my_proportion(self, player_idx):
        assassin_damage = 0.0
        for i in self.the_assassins:
            assassin_damage += self.attacks_by[i]

        return (self.attacks_by[player_idx] + 0.000001) / (assassin_damage + 0.000001)
