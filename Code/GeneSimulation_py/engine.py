import numpy as np
import math

class JHGEngine():
    def __init__(self, alpha=0.2, beta=0.75, give=1.4, keep=1.0, steal=1.1, num_players=3, base_popularity=100.0, poverty_line=0.0):
        # Scalars
        ## Weight the value of previous exchanges
        self.alpha = alpha

        ## Weights the value of previous popularity scores
        self.beta = beta

        ## scaling coefficient for tokens given, kept and stolen
        self.C_g = give
        self.C_k = keep
        self.C_s = steal

        ## total number of players
        self.N = num_players

        ## Popularity of player i at time t
        self.p0 = base_popularity # base_popularity can be either a scalar or a vector
        self.P = []
        self.P.append(np.ones(self.N) * self.p0)
        self.Ptemp = []
        self.Ptemp.append(np.ones(self.N) * self.p0)

        ## current round number
        self.t = 0

        ## historical transaction matrices
        self.T = []
        self.T.append(np.zeros((self.N, self.N)))

        ## influence matrices
        self.I = []
        self.I.append(np.eye(self.N)) # i = influences => j

        self.poverty_line = poverty_line

    def reset(self):
        ## Popularity of player i at time t
        self.P = []
        self.P.append(np.ones(self.N) * self.base_popularity)
        self.Ptemp = []
        self.Ptemp.append(np.ones(self.N) * self.base_popularity)

        ## current round number
        self.t = 0

        ## historical transaction matrices
        self.T = []
        self.T.append(np.zeros((self.N, self.N)))

        ## influence matrices
        self.I = []
        self.I.append(np.eye(self.N)) # i = influences => j

    def F(self, tau, t):
        self.I[tau] = self.I_(tau, t)
        # print("\nI[" + str(tau) + "]:")
        # print(self.I[tau])
        self.P[tau] = np.maximum(np.sum(self.I[tau], axis=0) + (1-self.alpha)**(tau) * self.p0, 0)

    def I_(self, tau, t):
        if tau <= 0:
            return np.zeros((self.N, self.N))
        I_bar = self.I_(tau - 1, t)     # this is Itemp[tau-]
        V = self.I_hat(tau, t)
        # I_tilde = self.I_hat(tau, t)    # this is the V

        # revised clipping method
        Itemp = self.alpha * V + (1 - self.alpha) * I_bar
        Itemp = self.scaleBackMurder(tau, Itemp, V)
        self.Ptemp[tau] = Itemp.sum(axis=0) +  + pow((1.0 - self.alpha), tau) * self.P[0]
        return Itemp
        
        # old clipping method
        # # Rescale the influence if the incoming negative influence is greater than the expected popularity
        # if np.any(I_tilde < 0):
        #     with np.errstate(divide='ignore', invalid='ignore'):
        #         omega = np.minimum(1, (self.P[tau - 1] + np.sum(self.alpha * np.abs(I_tilde * (I_tilde > 0)), axis=0) ) / np.sum(self.alpha * np.abs(I_tilde * (I_tilde < 0)), axis=0))
        #         omega[np.isnan(omega)] = 1.0
        #         omega[np.isinf(omega)] = 1.0
        #     omega = np.ones_like(I_tilde) * omega
        #     omega[I_tilde >= 0] = 1.0
        #     I_tilde = I_tilde * omega - np.diag(np.sum(np.abs(I_tilde * (1 - omega)), axis=1))

        # self.Ptemp[tau] = [ sum(y) for y in zip(*(self.alpha * I_tilde + (1 - self.alpha) * I_bar)) ] + pow((1.0 - self.alpha), tau) * self.P[0]
        # return self.alpha * I_tilde + (1 - self.alpha) * I_bar

        

    ## Raw round influence
    def I_hat(self, tau, t):
        T_ = self.T[tau]

        T_plus = np.clip(T_, 0, 1)
        np.fill_diagonal(T_plus, 0)

        T_minus = np.clip(T_, -1, 0)
        T_minus = T_minus * (self.P[tau - 1] > 0)
        np.fill_diagonal(T_minus, 0)

        T_diag = np.zeros_like(T_)
        np.fill_diagonal(T_diag, np.diag(T_))

        w = self.W(tau, t)

        # print("w: " + str(w))

        Cs = np.zeros_like(w)
        with np.errstate(divide='ignore', invalid='ignore'):
            Cs = self.C_s * np.maximum(1 - (np.diag(T_) * w) / (np.ones((1, self.N)) @ (np.abs(T_minus) * w[:, np.newaxis])), 0)
            Cs[np.isnan(Cs)] = 0.0
            Cs[np.isinf(Cs)] = 0.0

        pK = self.C_k * T_diag
        pG = self.C_g * T_plus
        pS = np.zeros_like(T_)
        np.fill_diagonal(pS, np.sum(-(Cs * T_minus), axis=1))

        pS += Cs * T_minus
        return w[:, np.newaxis] * (pK + pG + pS)    

    ## weight of player's actions at time tau based on i's populatiry at time tau - 1 and t - 1
    def W(self, tau, t):
        # experiment so that we don't ruin explode exponential in fame networks
        eta = sum(self.P[tau - 1]) / sum(self.P[t-1])
        # print("eta at " + str(tau) + ": " + str(eta))
        # eta = 1
        return self.beta * self.P[tau - 1] + (1 - self.beta) * eta * self.P[t - 1]

        # eta = sum(self.Ptemp[tau-1]) / sum(self.P[t-1])
        # return self.beta * self.Ptemp[tau - 1] + (1 - self.beta) * eta * self.P[t - 1]

    def scaleBackMurder(self, tau, Itemp, V):
        scaled_attack = np.ones(self.N, dtype=float)
        lost_benefit = np.zeros(self.N, dtype=float)

        gain = (Itemp.clip(0)).sum(axis=0) + (pow(1.0-self.alpha, tau) * self.P[0])
        loss = (np.negative(Itemp).clip(0)).sum(axis=0)
        taken = self.alpha * -1.0 * (np.negative(V).clip(0)).sum(axis=0)
        poverty_line_vec = self.poverty_line * np.ones(self.N, dtype=float)
        inthered = gain - poverty_line_vec - loss

        for i in range(0,self.N):
            if (inthered[i] < 0) and (taken[i] < 0):
                considered = inthered[i]
                if taken[i] > inthered[i]:
                    considered = taken[i]

                scaled_attack[i] = (taken[i] - inthered[i]) / taken[i]
                if scaled_attack[i] < 0.0:
                    scaled_attack[i] = 0.0

                for j in range(0,self.N):
                    if V[j][i] < 0.0:
                        perdido = V[j][i] - (scaled_attack[i] * V[j][i])
                        lost_benefit[j] -= self.alpha * perdido

        # print("gain: " + str(gain))
        # print("lost_benefit: " + str(lost_benefit))
        # print("scaled_attack: " + str(scaled_attack))

        for i in range(0,self.N):
            Itemp[i][i] -= lost_benefit[i]
            if Itemp[i][i] < 0.0:
                # print("tenemos una problema: " + V[i][i])
                # sys.exit()
                Itemp[i][i] = 0.0
            for j in range(0,self.N):
                if V[i][j] < 0.0:
                    Itemp[i][j] -= (1.0 - scaled_attack[j]) * V[i][j] * self.alpha
        
        return Itemp

    def setAlpha(self, new_alpha):
        self.alpha = new_alpha

    def setBeta(self, new_beta):
        self.beta = new_beta

    def setGive(self, new_give):
        self.give = new_give

    def setKeep(self, new_keep):
        self.keep = new_keep

    def setSteal(self, new_steal):
        self.steal = new_steal

    def setBasePop(self, new_base):
        self.base_popularity = new_base
        self.P[0] = np.ones(self.N) * self.base_popularity

    def apply_transaction(self, T):
        self.t += 1
        T = T / np.sum(np.abs(T), axis=1)[:, np.newaxis]
        self.T.append(T)
        self.I.append(np.zeros((self.N, self.N))) # i = influences => j
        self.P.append(np.zeros_like(self.P[-1]))
        self.Ptemp.append(np.zeros_like(self.P[-1]))
        self.F(self.t, self.t)
        # print("Ptemp:\n" + str(self.Ptemp))
        # print("P:\n" + str(self.P))

    def get_influence(self, t=None):
        return self.I[t] if t is not None else self.I[self.t]

    def get_prev_influence(self):
        if (self.t > 0):
            return self.I[self.t-1]
        else:
            return self.I[0]

    def get_popularity(self, t=None):
        return self.P[t] if t is not None else self.P[self.t]
    
    def get_transaction(self, t=None):
        return self.T[t] if t is not None else self.T[self.t]

    def step_back(self):
        if self.t > 0:
            self.T.pop()
            self.I.pop()
            self.P.pop()
            self.t -= 1

    def readGameFromFile(self, filepath):
        f = open(filepath, "r")

        line = f.readline()
        words = line.split(",")
        #print("len: " + str(len(words)))
        self.N = int(math.sqrt(len(words)-6))
        #print("numPlayers: " + str(numPlayers))

        self.P = []
        #self.P.append(np.ones(self.N) * self.p0)
        self.T = []
        #self.T.append(np.zeros((self.N, self.N)))
        for line in f:
            words = line.split(",")
            self.t = int(words[0])
            self.alpha = float(words[1])
            self.beta = float(words[2])
            self.C_g = float(words[3])
            self.C_k = float(words[4])
            self.C_s = float(words[5])

            # read in popularities
            tmp = np.ones(self.N)
            for i in range(0,self.N):
                tmp[i] = float(words[6+i])
            self.P.append(tmp)

            # read in transactions
            self.T.append(np.zeros((self.N, self.N)))
            c = 6 + self.N
            for i in range(0,self.N):
                for j in range(0,self.N):
                    self.T[self.t][i][j] = float(words[c])
                    c = c+1

            # print()
            # print("Round: " + str(self.t))
            # print(self.P[self.t])
            # print(self.T[self.t] * (2 * self.N))


        f.close()
