import numpy as np

def unpack (N, t, back_ptr):
    tags = [0 for x in xrange(N)]

    i = N
    while (i > 0):
        tags[i-1] = t
        t = back_ptr[t][i-1]
        i -= 1

    return tags


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is a size N array of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    emission_scores = np.array(emission_scores).T.tolist()

    # Score matrix L x N+1
    score = [[-np.inf for i in xrange(N+1)] for j in xrange(L)]

    # Back Pointer matrix L x N+1
    back_ptr = [[0 for i in xrange(N+1)] for j in xrange(L)]

    for i in xrange(0, L):
        score[i][0] = start_scores[i] + emission_scores[i][0]

    for i in xrange(1, N):
        for t in xrange(0, L):
            for t1 in xrange(0, L):
                tmp = score[t1][i-1] + trans_scores[t1][t]
                if tmp > score[t][i]:
                    score[t][i] = tmp
                    back_ptr[t][i] = t1
            score[t][i] += emission_scores[t][i]

    for i in xrange(0, L):
        score[i][N] = score[i][N-1] + end_scores[i]
        back_ptr[i][N] = i

    t_max = -np.inf
    vit_max = -np.inf

    for t in xrange(0, L):
        if score[t][N] > vit_max:
            t_max = t
            vit_max = score[t][N]

    return (vit_max, unpack(N, t_max, back_ptr))
