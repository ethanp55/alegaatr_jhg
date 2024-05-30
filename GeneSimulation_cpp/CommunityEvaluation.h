#ifndef COMMUNITYEVALUATION_H
#define COMMUNITYEVALUATION_H

using namespace std;

#include <iostream>
#include <string>
#include <set>

#include "GeneAgent.h"

class CommunityEvaluation {
public:
    CommunityEvaluation(set<int> _s, double _modularity, double _centrality, double _collectiveStrength, double _familiarity, double _prosocial) {
        for (set<int>::iterator itr = _s.begin(); itr != _s.end(); itr++)
            s.insert(*itr);

        modularity = _modularity;
        centrality = _centrality;
        collectiveStrength = _collectiveStrength;
        familiarity = _familiarity;
        prosocial = _prosocial;

        score = 0.0;
    }

    ~CommunityEvaluation() {}

    void computeScore(int *genes) {
        score = ((100-genes[GENE_w_modularity]) + (genes[GENE_w_modularity] * modularity)) / 100.0;
        score *= ((100-genes[GENE_w_centrality]) + (genes[GENE_w_centrality] * centrality)) / 100.0;
        score *= ((100-genes[GENE_w_collective_strength]) + (genes[GENE_w_collective_strength] * collectiveStrength)) / 100.0;
        score *= ((100-genes[GENE_w_familiarity]) + (genes[GENE_w_familiarity] * familiarity)) / 100.0;
        score *= ((100-genes[GENE_w_prosocial]) + (genes[GENE_w_prosocial] * prosocial)) / 100.0;

        // Change on May 12
        // REMOVING RANDOM
        // score += (rand() / ((double)RAND_MAX)) / 10000.0;    // random tie-breaking
    }

    string getString() {
        string str = "\n";
        str += "set: ";
        for (set<int>::iterator itr = s.begin(); itr != s.end(); itr++) {
            str += to_string(*itr) + " ";
        }
        str += "\n   modularity: " + to_string(modularity) + "\n";
        str += "   centrality: " + to_string(centrality) + "\n";
        str += "   collectiveStrength: " + to_string(collectiveStrength) + "\n";
        str += "   familiarity: " + to_string(familiarity) + "\n";
        str += "   prosocial: " + to_string(prosocial) + "\n";
        str += "   score: " + to_string(score) + "\n";

        return str;
    }

    string getString_short() {
        string str = "";
        for (set<int>::iterator itr = s.begin(); itr != s.end(); itr++) {
            str += to_string(*itr) + " ";
        }

        return str;
    }

    set<int> s;
    double modularity, centrality, collectiveStrength, familiarity, prosocial, score;
};

#endif