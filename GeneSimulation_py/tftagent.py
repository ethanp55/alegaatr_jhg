from GeneSimulation_py.baseagent import AbstractAgent
import random
import numpy as np
import sys
import math
import copy
from operator import itemgetter


class TFTAgent(AbstractAgent):

    def __init__(self, geneStr):  # Change on Sep 21
        super().__init__()
        self.whoami = "gene"
        self.count = 0
        self.relativeFitness = 0.0
        self.absoluteFitness = 0.0
        self.gameParams = {}
        self.is_initialized = False

        if geneStr == "":
            self.genes_long = []
            for i in range(0, self.num_gene_copies):
                gene_set = {
                    "initial_keep": np.random.randint(0, 101),
                    "initial_alloc_size": np.random.randint(0, 101),
                    "initial_perc_neg": np.random.randint(0, 101),
                    "match_type": np.random.randint(0, 101),
                    "not_enough_toks": np.random.randint(0, 101),
                    "too_many_toks": np.random.randint(0, 101),
                    "perc_keep_extra": np.random.randint(0, 101),
                }
                self.genes_long.append(gene_set)
        else:
            # read geneStr to set up the genotype
            words = geneStr.split("_")
            self.genes_long = []
            gene_set = {
                "initial_keep": int(words[1]),
                "initial_alloc_size": int(words[2]),
                "initial_perc_neg": int(words[3]),
                "match_type": int(words[4]),
                "not_enough_toks": int(words[5]),
                "too_many_toks": int(words[6]),
                "perc_keep_extra": int(words[7]),
            }
            self.genes_long.append(gene_set)

        self.theTracked = self.getTracked()
        # print(self.theTracked)
        self.played_genes = True

        # print(self.getString())

        # fp = open("../State/rnums.txt", "r")

        # self.randNums = []
        # for i in range(0,10000):
        #     self.randNums.append(int(fp.readline()))
        # self.randCount = 0

        # fp.close()

    def play_round(self, player_idx, round_num, received, popularities, influence, extra_data, v, transactions):
        num_players = len(popularities)
        num_tokens = num_players * 2

        if self.is_initialized == False:
            self.prev_popularities = np.zeros(num_players, dtype=float)
            self.profile = np.zeros(num_players, dtype=float)
            self.is_initialized = True

        # tokens_2_match = [3.5,   3.2,   0,    0,    0,    0.86, -1,    5.77,  0,    0.91]
        # print('starting')
        # print(self.allocate_2_profile(tokens_2_match, 10, 20))
        # print('ending')
        # sys.exit(1)

        # self.printT(player_idx, str(received))

        allocations = np.zeros(num_players, dtype=int)

        if self.theTracked != 99999:
            self.theTracked = self.getTracked()

        # if player_idx == self.theTracked:
        # if player_idx == 0:
        #     print()
        #     print("\n\nRound " + str(round_num) + " (Player " + str(self.theTracked) + ")")

        if round_num == 0:
            # print('initial_keep: ',self.genes_long[0]['initial_keep'])
            cuanto_guardo = int(num_tokens * (self.genes_long[0]['initial_keep'] / 100.0) + 0.5)

            tokens_remaining = num_tokens - cuanto_guardo
            todavia = set()
            for i in range(0, num_players):
                if (i != player_idx):
                    todavia.add(i)

            while (tokens_remaining > 0) and (len(todavia) > 0):
                noisy_initial_alloc = self.genes_long[0]['initial_alloc_size'] + random.randint(-10, 10)
                if noisy_initial_alloc > 100:
                    noisy_initial_alloc = 100
                if noisy_initial_alloc < 0:
                    noisy_initial_alloc = 0
                the_alloc_size = 1 + int((num_tokens - 1) * (noisy_initial_alloc / 100) + 0.5)
                if the_alloc_size > tokens_remaining:
                    the_alloc_size = tokens_remaining

                sel = random.choice(list(todavia))
                num = random.randint(0, 99)
                if num < self.genes_long[0]['initial_perc_neg']:
                    allocations[sel] = -the_alloc_size
                else:
                    allocations[sel] = the_alloc_size

                todavia.remove(sel)

                tokens_remaining = tokens_remaining - the_alloc_size

            allocations[player_idx] = cuanto_guardo + tokens_remaining
        else:
            tokens_2_match = np.zeros(num_players, dtype=float)
            sum = 0.0
            for i in range(0, num_players):
                if i == player_idx:
                    tokens_2_match[i] = 0
                elif self.genes_long[0]['match_type'] < 50:
                    # print('matcher')
                    tokens_2_match[i] = received[i] * num_tokens
                else:
                    if self.prev_popularities[player_idx] > 0.0:
                        tokens_2_match[i] = (self.prev_popularities[i] / self.prev_popularities[
                            player_idx]) * num_tokens * received[i]
                    else:
                        tokens_2_match[i] = 0.0
                sum += abs(tokens_2_match[i])

            # self.printT(player_idx, 'initial tokens_2_math' + str(tokens_2_match))
            # print('initial tokens_2_match', tokens_2_match)
            # print('sum = ', sum)
            # print('num_tokens = ', num_tokens)

            if sum == 0.0:
                # self.printT(player_idx, 'case 0.0')
                tokens_2_match[player_idx] = num_tokens
            elif sum > num_tokens:
                # self.printT(player_idx, 'case sum > num_tokens')
                if self.genes_long[0]['not_enough_toks'] < 50:
                    # print('not_enough_toks < 50')
                    for i in range(0, num_players):
                        tokens_2_match[i] = tokens_2_match[i] * (num_tokens / sum)
                else:
                    # print('not_enough_toks > 50')
                    # self.printT(player_idx, 'not_enough_toks >= 50')
                    impact = []
                    for i in range(0, num_players):
                        if i != player_idx:
                            amount = abs(influence[player_idx][i] + influence[i][player_idx]) / 2.0
                            impact.append([i, amount])

                    impact.sort(key=itemgetter(1), reverse=True)
                    # print(impact)

                    toks_remaining = num_tokens
                    for i in range(0, num_players - 1):
                        # print(impact[i][0])
                        over = toks_remaining - abs(tokens_2_match[impact[i][0]])

                        if over < 0:
                            if tokens_2_match[impact[i][0]] < 0:
                                tokens_2_match[impact[i][0]] = -toks_remaining
                            else:
                                tokens_2_match[impact[i][0]] = toks_remaining

                        toks_remaining = toks_remaining - abs(tokens_2_match[impact[i][0]])

            elif sum < num_tokens:
                # self.printT(player_idx, 'case sum < num_tokens')
                extra = (self.genes_long[0]['perc_keep_extra'] / 100.0) * (num_tokens - sum)

                if self.genes_long[0]['too_many_toks'] < 50:
                    # print('too many toks less than 50')
                    # self.printT(player_idx, 'less than 50')
                    toks_remaining = num_tokens - extra
                    for i in range(0, num_players):
                        if i == player_idx:
                            tokens_2_match[i] = extra
                        else:
                            tokens_2_match[i] = tokens_2_match[i] * (toks_remaining / sum)

                else:
                    # print('too many toks greater than 50')
                    # self.printT(player_idx, 'more than 50')
                    # self.printT(player_idx, extra)
                    # print(tokens_2_match)
                    tokens_2_match[player_idx] = extra
                    toks_remaining = num_tokens - extra - sum
                    # print(extra)
                    # print(toks_remaining)
                    # self.printT(player_idx, toks_remaining)

                    while toks_remaining > 0:
                        # self.printT(player_idx, toks_remaining)
                        the_alloc_size = 1.0
                        if the_alloc_size > toks_remaining:
                            the_alloc_size = toks_remaining

                        sel = random.randint(0, num_players - 1)
                        while sel == player_idx:
                            sel = random.randint(0, num_players - 1)

                        if tokens_2_match[sel] > 0.0:
                            tokens_2_match[sel] = tokens_2_match[sel] + the_alloc_size
                        elif tokens_2_match[sel] < 0.0:
                            tokens_2_match[sel] = tokens_2_match[sel] - the_alloc_size
                        else:
                            num = random.randint(0, 99)
                            if num < self.genes_long[0]['initial_perc_neg']:
                                tokens_2_match[sel] = -the_alloc_size
                            else:
                                tokens_2_match[sel] = tokens_2_match[sel] + the_alloc_size
                        toks_remaining = toks_remaining - the_alloc_size

                    # print(tokens_2_match)

            # print('ready to allocate')

            allocations = self.allocate_2_profile(tokens_2_match, num_players, num_tokens)

        for i in range(0, num_players):
            self.prev_popularities[i] = popularities[i]

        # do a simple check
        s = 0
        for i in range(0, num_players):
            s = s + abs(allocations[i])
        if s != num_tokens:
            print('wrong number of token allocations', s, num_tokens)
            print(allocations)
            sys.exit(1)

        # print(allocations)
        return allocations

    def getString(self):
        theStr = "tft"
        for key in self.genes_long[0]:
            theStr = theStr + "_" + str(self.genes_long[0][key])

        return theStr

    # [ 3.5   3.2   0.    0.    0.    0.86 -1.    5.77  0.    0.91]
    def allocate_2_profile(self, tokens_2_match, num_players, num_tokens):
        # print()
        # print(tokens_2_match)

        allocations = np.zeros(num_players, dtype=int)

        mag = 0.0
        dado = 0.0
        for i in range(0, num_players):
            allocations[i] = int(tokens_2_match[i])
            dado += abs(allocations[i])
            self.profile[i] = abs(tokens_2_match[i] - allocations[i])
            mag = mag + self.profile[i]

        # print(self.profile)
        # print(mag)
        # print(dado)
        # profile = [.5, .2, 0, 0, 0, .86, 0, .77, 0, .91]

        while (dado < num_tokens):
            num = random.randint(0, 100) / 100.0
            sum = 0.0
            for i in range(0, num_players):
                sum = sum + (self.profile[i] / mag)
                if num < sum:
                    # print('selected ' + str(i))
                    mag = mag - self.profile[i]
                    self.profile[i] = 0.0
                    if tokens_2_match[i] < 0:
                        allocations[i] = allocations[i] - 1
                    else:
                        allocations[i] = allocations[i] + 1
                    dado = dado + 1
                    break

        return allocations

    def printT(self, player_idx, s):
        if player_idx == self.theTracked:
            print(s)

    def getTracked(self):
        f = open("../GeneSimulation_py/ScenarioIndicator/theTracked.txt", "r")
        val = int(f.readline())
        f.close()

        return val

    def setGameParams(self, gameParams, _forcedRandom):
        self.gameParams = gameParams
        self.forced_random = _forcedRandom
