#ifndef ASSASSINAGENT_H
#define ASSASSINAGENT_H

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <fstream>

#include "AbstractAgent.h"

using namespace std;

class assassinAgent : public AbstractAgent {
public:
    assassinAgent() {
        cout << "assassinAgent" << endl;
        isInitialized = false;
        whoami = "assassin";
    }

    ~assassinAgent() {
        if (isInitialized) {
            delete[] attacksBy;
            delete[] givesBy;
            delete[] didNoNo;
        }
    }

    void playRound(int numPlayers, int numTokens, int playerIdx, int roundNum, double *received, double *popularities, double **influence, int *allocations) {
        for (int i = 0; i < numPlayers; i++) {
            allocations[i] = 0;
        }
            
        if (roundNum == 0) {
            initVars(numPlayers);

            allocations[playerIdx] = numTokens;
        }
        else {
            updateVars(numPlayers, playerIdx, influence);

            // decide how much to keep
            int keepTokens = 0;

            if (popularities[playerIdx] >= 0.5) {
                double attacked = attacksOnSelf(numPlayers, received, popularities);
                double w = 0.65;
                attacksOnMe = w * attacked + (1-w) * attacksOnMe;
                keepTokens = getMin((int)((attacksOnMe / popularities[playerIdx]) + 0.5), numTokens);

                // decide who to attack with remaining tokens
                int preyIdx = -1;
                double preyPop = 99999.0;
                double attackProportion = 0.0;
                double propAssassinAttack = getMyProportion(playerIdx);
                double attackPower = (popularities[playerIdx] / propAssassinAttack) * coefs[STEAL_IDX] * alpha;

                for (int i = 0; i < numPlayers; i++) {
                    if (theAssassins.find(i) == theAssassins.end()) {
                        if ((popularities[i] < preyPop) && (popularities[i] >= (attackPower / 2.0))) {
                            preyIdx = i;
                            preyPop = popularities[i];
                            attackProportion = getMin(popularities[i] / attackPower, 1.0);
                        }
                    }
                }
                
                // allocate keep and steal
                if (preyIdx >= 0) {
                    int stealTokens = (int)((attackProportion * 0.9) * (numTokens - keepTokens));
                    allocations[preyIdx] = -stealTokens;
                    allocations[playerIdx] = numTokens - stealTokens;
                }
                else {
                    allocations[playerIdx] = numTokens;
                }
            }
            else {
                allocations[playerIdx] = numTokens;
            }
        }

        // printf("Assassin %i allocation: ", playerIdx);
        // for (int i = 0; i < numPlayers; i++) {
        //     printf("%i ", allocations[i]);
        // }
        // printf("\n");
    }

    void postContract(int playerIdx) {}

private:
    bool isInitialized;
    set<int> theAssassins;
    double *attacksBy, *givesBy, attacksOnMe;
    bool *didNoNo;

    void initVars(int numPlayers) {
        attacksBy = new double[numPlayers];
        givesBy = new double[numPlayers];
        didNoNo = new bool[numPlayers];
        theAssassins.clear();
        for (int i = 0; i < numPlayers; i++) {
            attacksBy[i] = 0.0;
            givesBy[i] = 0.0;
            didNoNo[i] = false;
            theAssassins.insert(i);
        }

        attacksOnMe = 0.0;
    }

    void updateVars(int numPlayers, int playerIdx, double **influence) {
        // update the set of assassins
        for (int i = 0; i < numPlayers; i++) {
            attacksBy[i] = givesBy[i] = 0.0;
            didNoNo[i] = false;
            if (theAssassins.find(i) != theAssassins.end()) {
                for (int j = 0; j < numPlayers; j++) {
                    if (i == j)
                        continue;

                    if (influence[i][j] < 0.0) {
                        attacksBy[i] -= influence[i][j];
                        if (theAssassins.find(j) != theAssassins.end())
                            didNoNo[i] = true;
                    }
                    else if (influence[i][j] > 0.0) {
                        givesBy[i] += influence[i][j];
                        didNoNo[i] = true;
                    }
                }
            }
        }

        for (int i = 0; i < numPlayers; i++) {
            set<int>::iterator pli = theAssassins.find(i);
            if ((pli != theAssassins.end()) && (didNoNo[i] || ((attacksBy[playerIdx] > 0.0) && (attacksBy[i] == 0.0)))) {
                // printf("removing player %i from assassins list: %lf, %lf\n", i, givesBy[i], attacksBy[i]);
                theAssassins.erase(pli);
            }
        }

        // printf("assassins: ");
        // for (set<int>::iterator itr = theAssassins.begin(); itr != theAssassins.end(); itr++) {
        //     printf("%i ", *itr);
        // }
        // printf("\n");
    }

    double attacksOnSelf(int numPlayers, double *received, double *popularities) {
        double amount = 0.0;
        for (int i = 0; i < numPlayers; i++) {
            if (received[i] < 0)
                amount += received[i] * popularities[i];
        }

        return -amount;
    }

    double getMyProportion(int playerIdx) {
        double assassinDamage = 0.0;
        for (set<int>::iterator itr = theAssassins.begin(); itr != theAssassins.end(); itr++) {
            assassinDamage += attacksBy[*itr];
        }

        return ((attacksBy[playerIdx] + 0.000001) / (assassinDamage + 0.000001));
    }

    double getMin(double v1, double v2) {
        if (v1 < v2)
            return v1;
        return v2;
    }

};

#endif