#ifndef JHGENGINE_H
#define JHGENGINE_H

#include <iostream>

#include "AbstractAgent.h"

using namespace std;

class JHGEngine {
public:
    double coefs[3], alpha, beta, povertyLine;
    int numPlayers, numTokens, numRounds;
    double **P, **Ptemp, *Psum, ***T, ***I, ***Itemp, **V, ***X_pos, ***X_neg, *stealCoef;
    double *scaledAttack, *lostBenefit;
    int t;


    JHGEngine() {
        printf("JHGEngine default constructor called\n");
        exit(1);
    }

    JHGEngine(double *_coefs, double _alpha, double _beta, double _povertyLine, int _numPlayers, int _numTokens, int _numRounds, double *_initialPopularities) {
        for (int i = 0; i < 3; i++)
            coefs[i] = _coefs[i];
        alpha = _alpha;
        beta = _beta;
        numPlayers = _numPlayers;
        numTokens = _numTokens;
        numRounds = _numRounds;
        povertyLine = _povertyLine;

        t = 0;

        // allocate memory
        P = new double*[numRounds+2];
        Ptemp = new double*[numRounds+2];
        Psum = new double[numRounds+2];
        T = new double**[numRounds+2];
        I = new double**[numRounds+2];
        Itemp = new double**[numRounds+2];
        X_pos = new double**[numRounds+2];
        X_neg = new double**[numRounds+2];
        for (int i = 0; i < (numRounds+2); i++) {
            P[i] = new double[numPlayers];
            Ptemp[i] = new double[numPlayers];
            T[i] = new double*[numPlayers];
            I[i] = new double*[numPlayers];
            Itemp[i] = new double*[numPlayers];
            X_pos[i] = new double*[numPlayers];
            X_neg[i] = new double*[numPlayers];
            for (int j = 0; j < numPlayers; j++) {
                T[i][j] = new double[numPlayers];
                I[i][j] = new double[numPlayers];
                Itemp[i][j] = new double[numPlayers];
                X_pos[i][j] = new double[numPlayers];
                X_neg[i][j] = new double[numPlayers];
            }
        }
        V = new double*[numPlayers];
        for (int j = 0; j < numPlayers; j++)
            V[j] = new double[numPlayers];
        scaledAttack = new double[numPlayers];
        lostBenefit = new double[numPlayers];
        stealCoef = new double[numPlayers];      

        // initialize round 0 values
        Psum[0] = 0.0;
        for (int i = 0; i < numPlayers; i++) {
            Psum[0] += _initialPopularities[i];
            P[0][i] = _initialPopularities[i];
            Ptemp[0][i] = _initialPopularities[i];
            for (int j = 0; j < numPlayers; j++) {
                T[0][i][j] = 0.0;
                X_pos[0][i][j] = X_neg[0][i][j] = 0.0;

                if (i == j)
                    I[0][i][j] = 1.0;
                else
                    I[0][i][j] = 0.0;
                Itemp[0][i][j] = 0.0;
            }
        }
    }

    ~JHGEngine() {
        for (int i = 0; i < (numRounds+2); i++) {
            for (int j = 0; j < numPlayers; j++) {
                delete[] T[i][j];
                delete[] I[i][j];
                delete[] Itemp[i][j];
                delete[] X_pos[i][j];
                delete[] X_neg[i][j];
            }
            delete[] P[i];
            delete[] Ptemp[i];
            delete[] Itemp[i];
            delete[] T[i];
            delete[] I[i];
            delete[] X_pos[i];
            delete[] X_neg[i];
        }
        delete[] P;
        delete[] Ptemp;
        delete[] Itemp;
        delete[] Psum;
        delete[] T;
        delete[] I;
        delete[] X_pos;
        delete[] X_neg;

        for (int j = 0; j < numPlayers; j++)
            delete[] V[j];
        delete[] V;

        delete[] scaledAttack;
        delete[] lostBenefit;
        delete[] stealCoef;
    }

    void playRound(int **transactions) {
        t++;

        // printf("\nRound: %i\n", t);
        // printf("transactions:\n");
        // for (int k = 0; k < numPlayers; k++) {
        //     for (int m = 0; m < numPlayers; m++) {
        //         printf("%i ", transactions[k][m]);
        //     }
        //     printf("\n");
        // }
        // cout << endl;

        // store the transactions in T
        for (int i = 0; i < numPlayers; i++) {
            double smChck = 0.0;
            for (int j = 0; j < numPlayers; j++) {
                T[t][i][j] = ((double)(transactions[i][j])) / ((double)numTokens);
                if (transactions[i][j] >= 0) {
                    X_pos[t][i][j] = fabs(T[t][i][j]);
                    X_neg[t][i][j] = 0.0;
                }
                else {
                    X_neg[t][i][j] = fabs(T[t][i][j]);
                    X_pos[t][i][j] = 0.0;
                }

                smChck += fabs(T[t][i][j]);
            }

            if (fabs(smChck - 1.0) > 0.000001) {
                printf("something is wrong with token allocations: %lf (%i)\n", smChck, i);
                for (int k = 0; k < numPlayers; k++) {
                    for (int m = 0; m < numPlayers; m++) {
                        printf("%i ", transactions[k][m]);
                    }
                    printf("\n");
                }
                exit(1);
            }
        }

        // printf("T:\n");
        // for (int k = 0; k < numPlayers; k++) {
        //     for (int m = 0; m < numPlayers; m++) {
        //         printf("%.3lf ", T[t][k][m]);
        //     }
        //     printf("\n");
        // }
        // printf("\nX_pos:\n");
        // for (int k = 0; k < numPlayers; k++) {
        //     for (int m = 0; m < numPlayers; m++) {
        //         printf("%.3lf ", X_pos[t][k][m]);
        //     }
        //     printf("\n");
        // }
        // printf("\nX_neg:\n");
        // for (int k = 0; k < numPlayers; k++) {
        //     for (int m = 0; m < numPlayers; m++) {
        //         printf("%.3lf ", X_neg[t][k][m]);
        //     }
        //     printf("\n");
        // }
        // cout << endl;
        // exit(1);

        // update influences and popularities
        funcF(t, t);

        Psum[t] = 0.0;
        for (int i = 0; i < numPlayers; i++)
            Psum[t] += P[t][i];
        // printf("Ptemp:\n");
        // for (int tau = 0; tau < t+1; tau++) {
        //     for (int i = 0; i < numPlayers; i++) {
        //         printf("%.2lf ", Ptemp[tau][i]);
        //     }
        //     printf("\n");
        // }
        // if (t == numRounds) {
        //     printf("\nP:\n");
        //     for (int tau = 0; tau < t+1; tau++) {
        //         for (int i = 0; i < numPlayers; i++) {
        //             printf("%.2lf ", P[tau][i]);
        //         }
        //         printf("\n");
        //     }
        // }
    }

    void printP() {
        printf("\nP:\n");
        for (int tau = 0; tau < t+1; tau++) {
            for (int i = 0; i < numPlayers; i++) {
                printf("%.2lf ", P[tau][i]);
            }
            printf("\n");
        }
    }

    void funcF(int tau, int t) {
        // cout << "funcF" << endl;


        // update influences
        funcI(tau, t);
        for (int i = 0; i < numPlayers; i++) {
            for (int j = 0; j < numPlayers; j++) {
                I[tau][i][j] = Itemp[tau][i][j];
            }
            P[tau][i] = Ptemp[tau][i];
            if (P[tau][i] < 0.0)
                P[tau][i] = 0.0;
        }

        // printf("\nI[tau]:\n");
        // for (int i = 0; i < numPlayers; i++) {
        //     for (int j = 0; j < numPlayers; j++) {
        //         printf("%.2lf, ", I[tau][i][j]);
        //     }
        //     printf("\n");
        // }
    }

    void funcI(int tau, int t) {
        // cout << "funcI: " << i << ", " << j << endl;
        if (tau <= 0)
            return;

        funcI(tau-1, t);
        funcV(tau, t);

        // temporary computation
        for (int i = 0; i < numPlayers; i++) {
            for (int j = 0; j < numPlayers; j++) {
                Itemp[tau][i][j] = alpha * V[i][j] + (1.0 - alpha) * Itemp[tau-1][i][j];
            }
        }

        // printf("\n\nLet's go!!!\n");
        // printf("Itemp[%i]:\n", tau);
        // for (int i = 0; i < numPlayers; i++) {
        //     for (int j = 0; j < numPlayers; j++) {
        //         printf("%.2lf, ", (1.0 - alpha) * Itemp[tau-1][i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\nV[%i]:\n", tau);
        // for (int i = 0; i < numPlayers; i++) {
        //     for (int j = 0; j < numPlayers; j++) {
        //         printf("%.2lf, ", alpha * V[i][j]);
        //     }
        //     printf("\n");
        // }

        // now scale back stealing from the dead
        scaleBackMurder(tau);
        // for (int i = 0; i < numPlayers; i++) {
        //     for (int j = 0; j < numPlayers; j++) {
        //         Itemp[tau][i][j] = alpha * V[i][j] + (1.0 - alpha) * Itemp[tau-1][i][j];
        //     }
        // }
        for (int i = 0; i < numPlayers; i++)
            Ptemp[tau][i] = sumCol(Itemp[tau], i) + pow((1.0 - alpha), tau) * P[0][i];

        // printf("Ptemp[%i]: ", tau);
        // for (int i = 0; i < numPlayers; i++) {
        //     printf("%lf, ", Ptemp[tau][i]);
        // }
        // printf("\n");
        // printf("P[%i]: ", tau);
        // for (int i = 0; i < numPlayers; i++) {
        //     printf("%lf, ", P[tau][i]);
        // }
        // printf("\n");

    }

    void funcV(int tau, int t) {
        double v, w_i;

        // double stemp, s;
        // stemp = s = 0.0;
        // for (int i = 0; i < numPlayers; i++) {
        //     stemp += Ptemp[tau-1][i];
        //     s += P[t-1][i];
        // }
        // double eta = stemp / s;
        // double eta = Psum[tau-1] / Psum[t-1];   // term to account for inflation
        // printf("\neta at %i: %lf\n", tau, eta);

        for (int k = 0; k < numPlayers; k++)
            stealCoef[k] = funcSteal(tau, t, k);

        // printf("w: ");
        for (int i = 0; i < numPlayers; i++) {
            w_i = funcW(tau, t, i);
            // printf("%lf, ", w_i);
            for (int j = 0; j < numPlayers; j++) {
                v = 0.0;
                if (i == j) {
                    v = coefs[KEEP_IDX] * X_pos[tau][i][i];
                    for (int k = 0; k < numPlayers; k++) {
                        // v += funcSteal(tau, t, k) * X_neg[tau][i][k];
                        v += stealCoef[k] * X_neg[tau][i][k];
                    }
                }
                else {
                    v = coefs[GIVE_IDX] * X_pos[tau][i][j];
                    // v -= funcSteal(tau, t, j) * X_neg[tau][i][j];
                    v -= stealCoef[j] * X_neg[tau][i][j];
                }

                V[i][j] = v * w_i;
            }
        }
        // printf("\n");
    }

    double funcW(int tau, int t, int i) {
        // // cout << "funcW: " << tau << ", " << t << ", " << i << endl;

        // double stemp, s;
        // stemp = s = 0.0;
        // for (int i = 0; i < numPlayers; i++) {
        //     stemp += Ptemp[tau-1][i];
        //     s += P[t-1][i];
        // }
        // double eta = stemp / s;
        // return beta * Ptemp[tau-1][i] + (1.0 - beta) * eta * P[t-1][i];

        double eta = Psum[tau-1] / Psum[t-1];   // term to account for inflation
        return beta * P[tau-1][i] + (1.0 - beta) * eta * P[t-1][i];
    }

    double funcSteal(int tau, int t, int k) {
        double stoleFromK = 0;
        for (int j = 0; j < numPlayers; j++)
            stoleFromK += X_neg[tau][j][k] * funcW(tau, t, j);

        if (stoleFromK > 0.0) {
            double v = 1.0 - ((X_pos[tau][k][k] * funcW(tau, t, k)) / stoleFromK);
            if (v < 0.0)
                v = 0.0;

            return coefs[STEAL_IDX] * v;
        }
        else
            return coefs[STEAL_IDX];
    }

    void scaleBackMurder(int tau) {
        for (int i = 0; i < numPlayers; i++) {
            scaledAttack[i] = 1.0;
            lostBenefit[i] = 0.0;
        }

        // printf("Itemp (%i):\n", tau);
        // for (int k = 0; k < numPlayers; k++) {
        //     for (int m = 0; m < numPlayers; m++) {
        //         printf("%.3lf ", Itemp[tau][k][m]);
        //     }
        //     printf("\n");
        // }
        // printf("\nV:\n");
        // for (int k = 0; k < numPlayers; k++) {
        //     for (int m = 0; m < numPlayers; m++) {
        //         printf("%.3lf ", V[k][m]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        // printf("gain: ");

        double gain, loss, taken, inthered, considered;
        for (int i = 0; i < numPlayers; i++) {
            gain = sumSignedCol(Itemp[tau], i, true) + (pow(1-alpha, tau) * P[0][i]);
            loss = sumSignedCol(Itemp[tau], i, false);
            taken = alpha * -sumSignedCol(V, i, false);
            inthered = gain - povertyLine - loss;
            // printf("%lf ", gain);
            if ((inthered < 0) && (taken < 0)) {  // popularity below poverty line and stealing caused at least some of that
                // scale back total attacks
                // printf("player %i is below poverty at time %i (inthered = %lf; taken = %lf)\n", i, tau, inthered, taken);
                // considered = inthered;
                // if (taken > inthered)
                //     considered = taken;
                // scaledAttack[i] = 1.0 + (considered / loss);
                scaledAttack[i] = (taken - inthered) / taken;
                if (scaledAttack[i] < 0.0)
                    scaledAttack[i] = 0.0;

                // printf("%i (at %i): inthered: %lf; taken: %lf\n", i, tau, inthered, taken);
                // printf("\tscale back: %lf\n", scaledAttack[i]);

                for (int j = 0; j < numPlayers; j++) {
                    // if (Itemp[tau][j][i] < 0.0) {
                        // double perdido = Itemp[tau][j][i] - (scaledAttack[i] * Itemp[tau][j][i]);
                        // lostBenefit[j] -= perdido;
                    if (V[j][i] < 0.0) {
                        double perdido = V[j][i] - (scaledAttack[i] * V[j][i]);
                        // printf("\tattack amount lost by %i: %lf\n", j, alpha * perdido);    
                        lostBenefit[j] -= alpha * perdido;
                        // printf("%lf: %lf\n", scaledAttack[i] * Itemp[tau][j][i], perdido);
                    }
                }
            }
        }

        // printf("\n");
        // printf("lostBenefit: ");
        // for (int i = 0; i < numPlayers; i++)
        //     printf("%.2lf ", lostBenefit[i]);
        // printf("\nscaledAttack: ");
        // for (int i = 0; i < numPlayers; i++)
        //     printf("%.2lf ", scaledAttack[i]);
        // printf("\n");

        for (int i = 0; i < numPlayers; i++) {
            Itemp[tau][i][i] -= lostBenefit[i];
            if (Itemp[tau][i][i] < 0.0) {
                // printf("tenemos una problema: %lf\n", V[i][i]);
                // exit(1);
                Itemp[tau][i][i] = 0.0;
            }
            for (int j = 0; j < numPlayers; j++) {
                // if (Itemp[tau][i][j] < 0.0)
                //     Itemp[tau][i][j] = scaledAttack[j] * Itemp[tau][i][j];
                if (V[i][j] < 0.0) {
                    Itemp[tau][i][j] -= (1.0 - scaledAttack[j]) * V[i][j] * alpha;
                }
            }
        }

        // printf("\nnew Itemp (%i):\n", tau);
        // for (int k = 0; k < numPlayers; k++) {
        //     for (int m = 0; m < numPlayers; m++) {
        //         printf("%.3lf ", Itemp[tau][k][m]);
        //     }
        //     printf("\n");
        // }
        // printf("\n\n");
    }

    double sumSignedCol(double **x, int ind, bool positivo) {
        double s = 0.0;
        for (int i = 0; i < numPlayers; i++) {
            if (positivo && (x[i][ind] >= 0.0))
                s += x[i][ind];
            else if (!positivo && (x[i][ind] < 0.0))
                s += -x[i][ind];
        }
        
        return s;
    }

    double sumCol(double **x, int ind) {
        double s = 0.0;
        for (int i = 0; i < numPlayers; i++)
            s += x[i][ind];
        
        return s;
    }

    void save(string fnombre) {
        ofstream f(fnombre);

        // header
        f << "round,alpha,beta,give,keep,steal";
        for (int i = 0; i < numPlayers; i++)
            f << ",p" << i;
        for (int i = 0; i < numPlayers; i++) {
            for (int j = 0; j < numPlayers; j++) {
                f << "," << i << "-T-" << j;
            }
        }
        for (int i = 0; i < numPlayers; i++) {
            for (int j = 0; j < numPlayers; j++) {
                f << "," << i << "-I-" << j;
            }
        }
        f << endl;

        for (int tau = 0; tau <= t; tau++) {
            f << tau << "," << alpha << "," << beta << "," << coefs[1] << "," << coefs[0] << "," << coefs[2];

            // print popularity
            for (int i = 0; i < numPlayers; i++)
                f << "," << P[tau][i];

            // print transactions
            for (int i = 0; i < numPlayers; i++) {
                for (int j = 0; j < numPlayers; j++) {
                    f << "," << T[tau][i][j];
                }
            }

            // print influence
            for (int i = 0; i < numPlayers; i++) {
                for (int j = 0; j < numPlayers; j++) {
                    f << "," << I[tau][i][j];
                }
            }

            f << endl;
        }

        f.close();
    }

};

#endif