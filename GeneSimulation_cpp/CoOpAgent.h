#ifndef CO_OPAGENT_H
#define CO_OPAGENT_H

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <fstream>

#include "AbstractAgent.h"

using namespace std;

class CoOpAgent : public AbstractAgent {
public:
    double initialFeePercentage, feePercentage;
    string merit;

    CoOpAgent() {
        cout << "default CoOpAgent constructor" << endl;
        exit(1);
    }

    // coOp [initialFeePercentage] [feePercentage] [welfare=match/even]
    CoOpAgent(string line) {
        vector<string> words = parse(line);

        if (words.size() != 4) {
            cout << "WARNING: Default CoOp" << endl;
            initialFeePercentage = 100;
            feePercentage = 25;
            merit = "match";
        }
        else {
            initialFeePercentage = stof(words[1]);
            feePercentage = stof(words[2]);
            merit = words[3];
        }

        whoami = line;
        threshold = -5.0;
    }

    ~CoOpAgent() {
        if (isInitialized) {
            delete[] prevPopularities;
            delete[] tallyFees;
            delete[] tallyReceived;
            delete[] tallyUnpunished;
            delete[] tallyUnremunerated;
            delete[] membership;
            delete[] priorMembership;
            for (int i = 0; i < nPlayers; i++) {
                delete[] attacker[i];
                delete[] prevInfluence[i];
            }
            delete[] prevInfluence;
            delete[] attacker;
            delete[] owe;
            delete[] profile;
            delete[] cantidad;
        }
    }

    void playRound(int numPlayers, int numTokens, int playerIdx, int roundNum, double *received, double *popularities, double **influence, int *allocations) {
        // cout << "\nThe Co-Op" << endl;
        if (roundNum == 0) {
            initVars(numPlayers, playerIdx, popularities);
        }
        else {
            // determine new attacks (on self and members) -- essentially update debt to the CoOp
            determineAttackers(numPlayers, playerIdx, influence);

            updateTallies(numPlayers, numTokens, playerIdx, received, popularities, influence);
        }

        // update membership
        updateMembership(numPlayers, numTokens, playerIdx, received, popularities);

        allocateTokens(numPlayers, numTokens, playerIdx, roundNum, popularities, received, influence, allocations);

        // update some variables
        for (int i = 0; i < numPlayers; i++) {
            if ((i != playerIdx) && membership[i]) {
                if (allocations[i] > 0) {
                    double amount = (allocations[i] / ((double)numTokens)) * popularities[playerIdx] * coefs[GIVE_IDX] * alpha;
                    if (tallyUnremunerated[i] > 0.0) {
                        // cout << "remunerated to " << i << ": " << amount << endl;
                        tallyUnremunerated[i] -= amount;
                        // cout << "new tallyUnremunerated: " << tallyUnremunerated[i] << endl;
                        if (tallyUnremunerated[i] < 0.0) {
                            amount = -1.0 * tallyUnremunerated[i];
                            tallyUnremunerated[i] = 0.0;
                        }
                        else
                            amount = 0.0;
                        // cout << "  leftover: " << amount << endl;
                    }
                    tallyReceived[i] += amount;
                }
            }

            prevPopularities[i] = popularities[i];
            priorMembership[i] = membership[i];
            for (int j = 0; j < numPlayers; j++) {
                prevInfluence[i][j] = influence[i][j];
            }
        }

        // cout << "allocations: ";
        // for (int i = 0; i < numPlayers; i++) {
        //     cout << allocations[i] << " ";
        // }
        // cout << endl;
    }

    void postContract(int playerIdx) {
        // cout << "coOp postContract: " << playerIdx << endl;
        string fnombre = "Contracts/contract_" + to_string(playerIdx) + ".txt";
        ofstream output(fnombre);

        output << "initialFee: " << initialFeePercentage << endl;
        output << "fee: " << feePercentage << endl;
        output << "redistribution: " << merit << endl;

        output.close();
    }

private:
    double *prevPopularities, **prevInfluence, *tallyFees, *tallyReceived, *tallyUnpunished, *tallyUnremunerated, *owe, *profile;
    bool isInitialized, *membership, *priorMembership, **attacker;
    int nPlayers, *cantidad;
    double threshold, keepAmount;

    vector<string> parse(const string& s) {
        vector<string> tokens;
        string token;
        istringstream tokenStream(s);
        while (getline(tokenStream, token, ' '))
            tokens.push_back(token);
        return tokens;
    }

    void initVars(int numPlayers, int playerIdx, double *popularities) {
        if (!isInitialized) {
            tallyFees = new double[numPlayers];
            tallyReceived = new double[numPlayers];
            tallyUnpunished = new double[numPlayers];
            tallyUnremunerated = new double[numPlayers];
            prevPopularities = new double[numPlayers];
            membership = new bool[numPlayers];
            priorMembership = new bool[numPlayers];
            prevInfluence = new double*[numPlayers];
            attacker = new bool*[numPlayers];
            owe = new double[numPlayers];
            profile = new double[numPlayers];
            cantidad = new int[numPlayers];
        }
        for (int i = 0; i < numPlayers; i++) {
            tallyFees[i] = 0.0;
            tallyReceived[i] = 0.0;
            tallyUnpunished[i] = 0.0;
            tallyUnremunerated[i] = 0.0;
            prevPopularities[i] = 0.0;
            membership[i] = priorMembership[i] = false;
            if (!isInitialized) {
                prevInfluence[i] = new double[numPlayers];
                attacker[i] = new bool[numPlayers];
            }
            for (int j = 0; j < numPlayers; j++) {
                prevInfluence[i][j] = 0.0;
                attacker[i][j] = false;
            }
        }

        isInitialized = true;
        nPlayers = numPlayers;
        keepAmount = 0.5;//popularities[playerIdx] / 2.0;
    }

    void updateTallies(int numPlayers, int numTokens, int playerIdx, double *received, double *popularities, double **influence) {
        double pain;
        for (int i = 0; i < numPlayers; i++) {
            if (i == playerIdx)
                continue;

            if (membership[i]) {
                tallyFees[i] += received[i] * prevPopularities[i] * alpha;
            }

            for (int j = 0; j < numPlayers; j++) {
                if (attacker[i][j] && membership[j]) {
                    pain = (prevInfluence[i][j] * (1.0 - alpha)) - influence[i][j];
                    if ((tallyUnpunished[i] == 0.0) && (influence[j][i] < threshold)) {
                        // cout << "reducing pain from " << i << " to " << j << " by " << fabs(influence[j][i]) << endl;
                        pain += influence[j][i];
                    }
                    // cout << "pain of " << i << " on " << j << " is " << pain << endl;
                    if (pain > 0.0) {
                        tallyUnpunished[i] += pain;
                        tallyUnremunerated[j] += pain;
                    }
                }
            }

            if (popularities[i] <= povertyLine) {
                tallyUnpunished[i] = 0.0;
            }
            else if (influence[playerIdx][i] < threshold) {
                tallyUnpunished[i] -= getMax(prevInfluence[playerIdx][i] * (1.0 - alpha) - influence[playerIdx][i], 0.0);
                if (tallyUnpunished[i] < 0.0)
                    tallyUnpunished[i] = 0.0;
            }
        }

        // cout << "tallyFees: ";
        // for (int i = 0; i < numPlayers; i++) {
        //     cout << tallyFees[i] << " ";
        // }
        // cout << endl;
        // cout << "tallyReceived: ";
        // for (int i = 0; i < numPlayers; i++) {
        //     cout << tallyReceived[i] << " ";
        // }
        // cout << endl;
        // cout << "tallyUnremunerated: ";
        // for (int i = 0; i < numPlayers; i++) {
        //     cout << tallyUnremunerated[i] << " ";
        // }
        // cout << endl;
        // cout << "tallyUnpunished: ";
        // for (int i = 0; i < numPlayers; i++) {
        //     cout << tallyUnpunished[i] << " ";
        // }
        // cout << endl;

    }

    void updateMembership(int numPlayers, int numTokens, int playerIdx, double *received, double *popularities) {
        // cout << "membership: ";
        double proportion;
        for (int i = 0; i < numPlayers; i++) {
            if (i == playerIdx) {
                membership[i] = true;
            }
            else {
                proportion = 100.0 * received[i];
                // cout << i << ": " << proportion << " vs " << initialFeePercentage << endl;
                if (membership[i]) {
                    if ((proportion < feePercentage) || memberEnemy(i, numPlayers))
                        membership[i] = false;      // membership revoked
                }
                else if ((proportion >= initialFeePercentage) && !memberEnemy(i, numPlayers)) {
                    membership[i] = true;

                    tallyFees[i] += (feePercentage / 100.0) * popularities[i] * alpha; // give them credit for normal fee payment
                }
            }
            // cout << (int)membership[i] << " ";
        }
        // cout << endl;
    }

    void determineAttackers(int numPlayers, int playerIdx, double **influence) {
        for (int i = 0; i < numPlayers; i++) {
            if (i == playerIdx)
                continue;
            
            for (int j = 0; j < numPlayers; j++) {
                if ((influence[i][j] < threshold) && (prevInfluence[i][j] > threshold) && membership[j]) {
                    // cout << i << " is an attacker of " << j << endl;
                    attacker[i][j] = true;
                }
                if ((influence[i][j] >= threshold) && attacker[i][j]) {
                    attacker[i][j] = false;
                    // cout << i << " is no longer an attacker of " << j << endl;
                }
            }
        }
    }

    bool memberEnemy(int index, int numPlayers) {
        for (int i = 0; i < numPlayers; i++) {
            if (attacker[index][i] && priorMembership[i])
                return true;
        }

        return false;
    }

    // determine token allocations
    // Strategy:
    //    Priority 1) Defend self from attacks
    //    Priority 2) Attack offenders (with great strength)
    //    Priority 3) Restore wealth of member victims
    //    Priority 4) Reimburse membership fees (as much as possible)
    //    Priority 5) Give extra dividends (when available and sufficiently strong)
    void allocateTokens(int numPlayers, int numTokens, int playerIdx, int roundNum, double *popularities, double *received, double **influence, int *allocations) {
        for (int i = 0; i < numPlayers; i++)
            allocations[i] = 0;

        int numToksRemaining = numTokens;

        int numDefend = defendSelf(numPlayers, numTokens, playerIdx, roundNum, popularities, received, allocations);
        // cout << "numDefend: " << numDefend << endl;
        numToksRemaining -= numDefend;

        int numAttacks = attackUnpunished(numPlayers, numTokens, numToksRemaining, playerIdx, popularities, influence, allocations);
        // cout << "numAttacks: " << numAttacks << endl;
        numToksRemaining -= numAttacks;

        int numRemunerate = remunerateDamage(numPlayers, numTokens, numToksRemaining, playerIdx, popularities, allocations);
        // cout << "numRemunerate: " << numRemunerate << endl;
        numToksRemaining -= numRemunerate;

        int numReimburse = reimburseMemberFees(numPlayers, numTokens, numToksRemaining, playerIdx, popularities, allocations);
        // cout << "numReimburse: " << numReimburse << endl;
        numToksRemaining -= numReimburse;

        int numBenefits = disperseMemberBenefits(numPlayers, numTokens, numToksRemaining, playerIdx, popularities, allocations);
        // cout << "numBenefits: " << numBenefits << endl;
        numToksRemaining -= numBenefits;


        int dado = 0;
        for (int i = 0; i < numPlayers; i++) {
            dado += abs(allocations[i]);
        }
        if (dado > numTokens) {
            cout << "demasiado tokens allocated" << endl;
            exit(1);
        }
        allocations[playerIdx] += (numTokens - dado);
    }

    int defendSelf(int numPlayers, int numTokens, int playerIdx, int roundNum, double *popularities, double *received, int *allocations) {
        if (popularities[playerIdx] <= 0.0)
            return 0.0;

        if (roundNum > 0) {
            double lastAttacks = 0.0;
            // double mostPopularOther = 0.0;
            for (int i = 0; i < numPlayers; i++) {
                if (i == playerIdx)
                    continue;

                if (received[i] < 0.0)
                    lastAttacks += fabs(received[i]) * popularities[i];

                // if (popularities[i] > mostPopularOther)
                //     mostPopularOther = popularities[i];
            }

            // cout << "mostPopularOther: " << mostPopularOther << endl;

            // double fearAmount = (((coefs[STEAL_IDX] - 1.0) * mostPopularOther) / (coefs[STEAL_IDX] * popularities[playerIdx]));
            // // cout << "fearAmount: " << fearAmount << endl;
            // double MAX_FEAR = 0.5;
            // if (fearAmount > MAX_FEAR)
            //     fearAmount = MAX_FEAR;

            // cout << "lastAttacks: " << lastAttacks << endl;

            double attackAmount = lastAttacks / popularities[playerIdx];
            // cout << "attackAmount: " << attackAmount << endl;

            // double amount = getMax(attackAmount, fearAmount);
            double lambda;
            if (attackAmount > keepAmount)
                lambda = 0.25;
            else
                lambda = 0.75;
            keepAmount = lambda * keepAmount + (1.0 - lambda) * attackAmount;
            // cout << "keepAmount: " << keepAmount << endl;

            if (keepAmount < 0.0) {
                cout << "negative amount" << endl;
                exit(1);
            }
            else if (keepAmount > 1.0) {
                keepAmount = 1.0;
            }
        }

        allocations[playerIdx] = (int)((keepAmount * numTokens) + 0.5);

        return allocations[playerIdx];
    }

    int attackUnpunished(int numPlayers, int numTokens, int numToksRemaining, int playerIdx, double *popularities, double **influence, int *allocations) {
        if (numToksRemaining <= 0)
            return 0;

        // cout << "Considering attacks:"  << endl;
        double mag = 0.0;
        for (int i = 0; i < numPlayers; i++) {
            if (tallyUnpunished[i] > 0.0) {
                // decide how many tokens to use in the attack and what the value of the attack will be
                double punishAmount = getMin(tallyUnpunished[i], popularities[i] - povertyLine);
                double theyKeep = isKeeping(i, numPlayers, influence);
                double keepingTokens = theyKeep * (popularities[i] / popularities[playerIdx]);
                double tokensNeeded = punishAmount / (popularities[playerIdx] * coefs[STEAL_IDX] * alpha);
                cantidad[i] = (int)(1.5 * (tokensNeeded + keepingTokens) * numTokens + 0.5);
                double cantidadRatio = ((double)(cantidad[i])) / numTokens;

                // cout << i << " isKeeping: " << theyKeep << endl;
                // cout << "  attack " << i << " with " << cantidad[i] << endl;

                if (cantidad[i] > numToksRemaining) {
                    cantidad[i] = numToksRemaining;
                    cantidadRatio = 1.0;
                }

                // cout << "  (revised) attack " << i << " with " << cantidad[i] << endl;

                double gain = getMax((popularities[playerIdx] * cantidadRatio) - (popularities[i] * theyKeep), 0.0);
                double stealROI = (gain * coefs[STEAL_IDX]) / (cantidadRatio * popularities[playerIdx]);
                double damage = gain * coefs[STEAL_IDX] * alpha;
                double immGain = (stealROI - coefs[KEEP_IDX]) * cantidadRatio * popularities[playerIdx] * alpha;
                double vengenceAdvantage = immGain + damage;

                // cout << "  damage: " << damage << endl;
                // cout << "  immGain: " << immGain << endl;
                // cout << "  vengenceAdvantage: " << vengenceAdvantage << endl;

                if (vengenceAdvantage > 0.0) {
                    double valor = (((((double)cantidad[i]) / numTokens) - keepingTokens)) * popularities[playerIdx];
                    owe[i] = valor / cantidad[i];
                }
                else
                    owe[i] = 0.0;

                // cout << "  valor: " << owe[i] << endl;
            }
            else {
                owe[i] = 0.0;
                cantidad[i] = 0;
            }
            mag += owe[i];
        }

        int numAttackTokens = 0;
        if (mag > 0.0) {
            for (int i = 0; i < numPlayers; i++) {
                profile[i] = owe[i] / mag;
            }

            int sel =  selectFromProfile(profile, numPlayers);
            allocations[sel] = -cantidad[sel];
            numAttackTokens = -allocations[sel];
        }

        return numAttackTokens;
    }

    int remunerateDamage(int numPlayers, int numTokens, int numToksRemaining, int playerIdx, double *popularities, int *allocations) {
        if (numToksRemaining <= 0)
            return 0;

        double mag = 0.0;
        for (int i = 0; i < numPlayers; i++) {
            if (membership[i]) {
                owe[i] = tallyUnremunerated[i];
                mag += owe[i];
            }
            else
                owe[i] = 0.0;
        }

        int numRemunerate = 0;
        while ((mag >= 0.5) && ((numToksRemaining - numRemunerate) > 0)) {
            for (int i = 0; i < numPlayers; i++) {
                profile[i] = owe[i] / mag;
            }

            int sel =  selectFromProfile(profile, numPlayers);
            allocations[sel] ++;
            numRemunerate ++;
            double old = owe[sel];
            owe[sel] = getMax(owe[sel] - (coefs[GIVE_IDX] * (popularities[playerIdx] / numTokens) * alpha), 0.0);
            mag -= old - owe[sel];
        }

        return numRemunerate;
    }

    int reimburseMemberFees(int numPlayers, int numTokens, int numToksRemaining, int playerIdx, double *popularities, int *allocations) {
        // cout << "reimburseMemberFees" << endl;
        if (numToksRemaining <= 0)
            return 0;
        
        double mag = 0.0;
        for (int i = 0; i < numPlayers; i++) {
            if (membership[i]) {
                owe[i] = getMax(tallyFees[i] - tallyReceived[i], 0.0);
                mag += owe[i];
            }
            else
                owe[i] = 0.0;
        }

        int numReimburse = 0;
        while ((mag >= 0.5) && ((numToksRemaining - numReimburse) > 0)) {
            for (int i = 0; i < numPlayers; i++) {
                profile[i] = owe[i] / mag;
            }

            // cout << "mag: " << mag << endl;

            int sel =  selectFromProfile(profile, numPlayers);
            allocations[sel] ++;
            numReimburse ++;
            double old = owe[sel];
            owe[sel] = getMax(owe[sel] - (coefs[GIVE_IDX] * (popularities[playerIdx] / numTokens) * alpha), 0.0);
            mag -= old - owe[sel];
        }

        return numReimburse;
    }

    int disperseMemberBenefits(int numPlayers, int numTokens, int numToksRemaining, int playerIdx, double *popularities, int *allocations) {
        if (numToksRemaining <= 0)
            return 0;

        double mostPopular = 0.0;
        for (int i = 0; i < numPlayers; i++) {
            if (i == playerIdx)
                continue;

            if (mostPopular < popularities[i])
                mostPopular = popularities[i];
        }

        double mult = coefs[GIVE_IDX];
        // if (popularities[playerIdx] >= mostPopular)
        //     mult = coefs[GIVE_IDX];
        // else if (popularities[playerIdx] < (mostPopular / 2.0))
        //     mult = 1.0;
        // else {
        //     mult = (coefs[GIVE_IDX] - 1.0) * ((popularities[playerIdx] - (mostPopular / 2.0)) / (mostPopular - (mostPopular / 2.0)));
        //     mult += 1.0;
        // }
        // cout << "mult: " << mult << endl;
        
        // cout << "owe: ";
        double mag = 0.0;
        for (int i = 0; i < numPlayers; i++) {
            if (membership[i] && (i != playerIdx)) {
                double already = (allocations[i] / ((double)numTokens)) * popularities[playerIdx] * coefs[GIVE_IDX] * alpha;
                owe[i] = getMax(tallyFees[i] * mult - (tallyReceived[i] + already), 0.0);
                mag += owe[i];
                // cout << "(" << already << ") ";
            }
            else
                owe[i] = 0.0;
            // cout << owe[i] << " ";
        }
        // cout << endl;

        int numBenefits = 0;
        while ((mag >= 0.5) && ((numToksRemaining - numBenefits) > 0)) {
            for (int i = 0; i < numPlayers; i++) {
                profile[i] = owe[i] / mag;
            }

            int sel =  selectFromProfile(profile, numPlayers);
            allocations[sel] ++;
            numBenefits ++;
            double old = owe[sel];
            owe[sel] = getMax(owe[sel] - (coefs[GIVE_IDX] * (popularities[playerIdx] / numTokens) * alpha), 0.0);
            mag -= old - owe[sel];
        }

        return numBenefits;
    }

    int selectFromProfile(double *profile, int numPlayers) {
        double num = rand() / ((double)RAND_MAX);
        double s = 0.0;

        for (int i = 0; i < numPlayers; i++) {
            s += profile[i];
            if ((num <= s) && (s != 0.0))
                return i;
        }

        cout << "Problem: Didn't selectFromProfile" << endl;
        cout << "num: " << num << "; s: " << s << endl;
        cout << "profile: ";
        for (int i = 0; i < numPlayers; i++) {
            cout << profile[i] << " ";
        }
        cout << endl;
        exit(-1);

        return -1;
    }

    double isKeeping(int otherIdx, int numPlayers, double **influence) {
        double meAmount = 0.0;
        double totalAmount = 0.0;
        for (int i = 0; i < numPlayers; i++) {
            if (i == otherIdx)
                continue;

            if (influence[otherIdx][i] < 0.0) {
                totalAmount += -influence[otherIdx][i] / coefs[STEAL_IDX];
                meAmount -= -influence[otherIdx][i];
            }
            else {
                totalAmount += influence[otherIdx][i] / coefs[GIVE_IDX];
            }
        }

        meAmount = (meAmount + influence[otherIdx][otherIdx]) / coefs[KEEP_IDX];
        totalAmount += meAmount;

        if (totalAmount > 0.0)
            return meAmount / totalAmount;
        else
            return 1.0;
    }
    

    double getMax(double v1, double v2) {
        if (v1 > v2)
            return v1;
        return v2;
    }

    double getMin(double v1, double v2) {
        if (v1 < v2)
            return v1;
        return v2;
    }

};

#endif