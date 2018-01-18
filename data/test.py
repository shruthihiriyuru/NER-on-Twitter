#!/bin/python
import numpy as np

def run_viterbi_test():
    """A simple tester for Viterbi algorithm.

    This function generates a bunch of random emission and transition scores,
    and computes the best sequence by performing a brute force search over all
    possible sequences and scoring them. It then runs Viterbi code to see what
    is the score and sequence returned by it.

    Compares both the best sequence and its score to make sure Viterbi is correct.
    """
    from viterbi import run_viterbi
    from numpy import random
    from itertools import product

    maxN = 7 # maximum length of a sentence (min is 1)
    maxL = 4 # maximum number of labels (min is 2)
    num_tests = 1 # number of sentences to generate
    random.seed(0)
    tolerance = 1e-5 # how close do the scores have to be?

    emission_var = 1.0 # variance of the gaussian generating emission scores
    trans_var = 1.0 # variance of the gaussian generating transition scores

    passed_y = 0 # how many times the correct sequence was predicted
    passed_s = 0 # how many times the correct score was returned

    for t in xrange(num_tests):
        N = 2
        L = 3

        # Generate the scores
        # emission_scores = random.normal(0.0, emission_var, (N,L))
        # trans_scores = random.normal(0.0, trans_var, (L,L))
        # start_scores = random.normal(0.0, trans_var, L)
        # end_scores = random.normal(0.0, trans_var, L)

        #print start_scores

        emission_scores = np.array([[0.1, 0.3, 0.25], [0.2, 0.45, 0.31]])
        trans_scores = np.array([[0.3, 0.2, 0.5], [0.12, 0.4, 0.3], [0.1, 0.6, 0.5]])
        start_scores = np.array([0.2, 0.5, 0.7])
        end_scores = np.array([0.5, 0.4, 0.1])

        # run viterbi
        (viterbi_s,viterbi_y) = run_viterbi(emission_scores, trans_scores, start_scores, end_scores)
        print "Viterbi", viterbi_s, viterbi_y

        # compute the best sequence and score
        best_y = []
        best_s = -np.inf
        for y in product(range(L), repeat=N): # all possible ys
            # compute its score
            score = 0.0
            score += start_scores[y[0]]
            for i in xrange(N-1):
                score += trans_scores[y[i], y[i+1]]
                score += emission_scores[i,y[i]]
            score += emission_scores[N-1,y[N-1]]
            score += end_scores[y[N-1]]
            # update the best
            if score > best_s:
                best_s = score
                best_y = list(y)
        print "Brute", best_s, best_y

        # mismatch if any label prediction doesn't match
        match_y = True
        for i in xrange(len(best_y)):
            if viterbi_y[i] != best_y[i]:
                match_y = False
        if match_y: passed_y += 1
        # the scores should also be very close
        print "scores: "
        print viterbi_s, best_s
        if abs(viterbi_s-best_s) < tolerance:
            passed_s += 1

    print "Passed(y)", passed_y*100.0/num_tests
    print "Passed(s)", passed_s*100.0/num_tests
    assert passed_y == num_tests
    assert passed_s == num_tests

if __name__ == "__main__":
    run_viterbi_test()



# import numpy as np
# from viterbi import run_viterbi
#
# em_s = np.array([[0.1, 0.3, 0.25], [0.2, 0.45, 0.31]])
# t_s = np.array([[0.3,0.2,0.5], [0.12,0.4,0.3], [0.1, 0.6,0.5]])
# s_s = np.array([0.2,0.5,0.7])
# en_s = np.array([0.5,0.4,0.1])
#
# print run_viterbi(em_s, t_s, s_s, en_s)
