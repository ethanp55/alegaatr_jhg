#ifndef DEFS_H
#define DEFS_H

#include <iostream>

#define KEEP_IDX   0
#define GIVE_IDX    1
#define STEAL_IDX    2

#define predef_NUMGENES     33

#define GENE_visualTrait            0   // unused
#define GENE_homophily              1   // unused
#define GENE_alpha                  2   
#define GENE_otherishDebtLimits     3
#define GENE_coalitionTarget        4
#define GENE_fixedUsage             5
#define GENE_w_modularity           6
#define GENE_w_centrality           7
#define GENE_w_collective_strength  8
#define GENE_w_familiarity          9
#define GENE_w_prosocial            10
#define GENE_initialDefense         11
#define GENE_minKeep                12
#define GENE_defenseUpdate          13
#define GENE_defensePropensity      14
#define GENE_fearDefense            15
#define GENE_safetyFirst            16
#define GENE_pillageFury            17
#define GENE_pillageDelay           18
#define GENE_pillagePriority        19
#define GENE_pillageMargin          20
#define GENE_pillageCompanionship   21
#define GENE_pillageFriends         22
#define GENE_vengenceMultiplier     23
#define GENE_vengenceMax            24
#define GENE_vengencePriority       25
#define GENE_defendFriendMultiplier 26
#define GENE_defendFriendMax        27
#define GENE_defendFriendPriority   28
#define GENE_attackGoodGuys         29
#define GENE_limitingGive           30
#define GENE_groupAware             31
#define GENE_joinCoop               32  // used only when a Co-op is specified (not part of the IJCAI 2024 paper)

#endif