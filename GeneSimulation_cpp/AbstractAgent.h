#ifndef ABSTRACTAGENT_H
#define ABSTRACTAGENT_H

#include <iostream>
#include "defs.h"

class AbstractAgent {
public:
    AbstractAgent() {}
    ~AbstractAgent() {}

    virtual void playRound(int numPlayers, int numTokens, int playerIdx, int roundNum, double *received, double *popularities, double **influence, int *allocations) = 0;

    virtual void postContract(int playerIdx) = 0;

    void setGameParams(double _coefs[3], double _alpha, double _beta, double _povertyLine, bool _forcedRandom) {
        alpha = _alpha;
        beta = _beta;
        povertyLine = _povertyLine;
        for (int i = 0; i < 3; i++)
            coefs[i] = _coefs[i];
        forcedRandom = _forcedRandom;
    }

    double coefs[3], alpha, beta, povertyLine;
    bool forcedRandom;
    std::string whoami;
};

#endif