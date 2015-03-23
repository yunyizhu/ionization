#!/usr/bin/python
import numpy as np
from math import exp
import scipy.weave as weave
import scipy.sparse as sp
from itertools import product
import pdb



class seq_to_feature:
    def __init__(self,  n):
        self.n = n
        self.nMers = []
        self.nMer_dict = {}
        
    def get_nMers(self,  seqs):
        tmp_nMers = []
        for seq in seqs:
            for i in range(0,  len(seq)-self.n+1):
                tmp_nMers.append( seq[i:(i+self.n)].upper() )
        self.nMers = list( set(tmp_nMers) )
        self.nMer_dict = {self.nMers[i]:i for i in range(len(self.nMers))}
        return self.nMers
    
    def construct_nMers(self,  AAs):
        tmp_nMers = itertools.product( AAs,  repeat = self.n )
        self.nMers = [''.join(nMer) for nMer in tmp_nMers]
        self.nMer_dict = {self.nMers[i]:i for i in range(len(self.nMers))}
        return self.nMers
    
    def BoW(self,  seqs,  normed = 1):
        #normed = 0: do not normalize, =1: normalize by sum, =2: normalize by maximum
        M = len(seqs)
        N = len(self.nMers)
        x = sp.lil_matrix((M,  N),  dtype = 'float64')
        
        for i in range(M):
            for j in range(len(seqs[i])-self.n+1):
                nMer = seqs[i][j:(j+self.n)].upper()
                x[i, self.nMer_dict[nMer]] += 1
            if normed == 1:
                x[i, : ] = x[i,:]*1.0/x[i,:].sum()
            elif normed == 2:
                x[i, : ] = x[i,:]*1.0/x[i,:].max()
        return x
    
    def alignment(self,  seqs,  AA_dict,   paras,  para_ind = [],  show_alignment = 0,  normed = 2):
        M = len(seqs)
        N = len(self.nMers)
        X = np.zeros(( M,  N ))

        if show_alignment == 2:
            p = []
            for i in range(M):
                pi = []
                for j in range(N):
                    X[i,  j],  pj = local_align(seqs[i],  self.nMers[j],   AA_dict,  paras,  para_ind,  show_alignment)
                    pi.append(pj)
                p.append(pi[:])
            if normed == 1:
                denominator = np.sum(X,  -1)[:, np.newaxis]
                denominator[ denominator ==0 ] = 0.1
            elif normed == 2:
                denominator = np.max(X,  -1)[:, np.newaxis]
                denominator[ denominator ==0 ] = 0.1
            else:
                denominator = 1.0
            return X*1.0/denominator,  np.array(p)

        else:
            for i in range(M):
                for j in range(N):
                    X[i,  j] = kernel.local_align(seqs[i],  self.nMers[j],  AA_dict,  paras,  para_ind,  show_alignment)
            
            if normed == 1:
                denominator = np.sum(X,  -1)[:, np.newaxis]
                denominator[ denominator ==0 ] = 0.1
            elif normed == 2:
                denominator = np.max(X,  -1)[:, np.newaxis]
                denominator[ denominator ==0 ] = 0.1
            else:
                denominator = 1.0
            return X*1.0/denominator

def f_select(x,  n_f, f_type):
    tot_s,  tot_f = np.shape(x)
    if f_type==1:
        f_score = np.reshape( np.array( x.sum(axis=0) ), tot_f )

    elif f_type==2:
        f_score = np.zeros(tot_f)
        for i in range(tot_s):
            f_sort = np.argsort( np.reshape( x[i, :].toarray() , tot_f))[::-1]
            f_score[ f_sort[:n_f] ] += 1

    f_sort = np.argsort(f_score)[::-1]
    return f_sort[:n_f]

def local_align(seqRow,  seqCol,  AA_dict,  paras,  para_ind = [],  show_alignment = 0):
    #initialize
    sub_mat = paras[:-1]
    gap_penalty = paras[-1]
    para_dic = dict()
    for i, para in enumerate(para_ind):
        para_dic[para] = i
    nRow = len(seqRow) + 1
    nCol = len(seqCol) + 1
    score_mat = np.zeros((nRow,  nCol))
    path = -1*np.ones((nRow ,  nCol)) # -1: new alignment   -2:from left -3:from up  >=0: index in the sub_mat matrix
    seqr = np.array([AA_dict[i] for i in seqRow.upper() ])
    seqc = np.array([AA_dict[i] for i in seqCol.upper() ])
    l,  gap_ind = len(AA_dict),  len(sub_mat)
    
    #update score matrix and path
    code="""
    for (int i=1; i<nRow; i++)
    {
    for(int j=1; j<nCol; j++)
    {
    double f[4] = {0, 0, 0, 0};
    int rn = seqr[i-1];
    int cn = seqc[j-1];
    if (rn>cn)
    {int tmp = rn; rn=cn; cn=tmp;}
    int ind = (2*l-rn+1)*rn/2 + cn - rn;
    f[1] = score_mat[ (i-1)*nCol +  j-1 ] + sub_mat[ind];
    f[2] = score_mat[ i*nCol + j-1 ] + gap_penalty;
    f[3] = score_mat[ (i-1)*nCol + j ] + gap_penalty;
    double max_score = 0;
    int direction[4] = {-1, ind, -2, -3};
    int p = -1;
    for (int k=1; k<4; k++)
    if (f[k]>max_score)
    {p = direction[k]; max_score = f[k];}
    score_mat[i*nCol + j] = max_score;
    path[i*nCol + j] = p;
    }
    }"""
    weave.inline(code,  ['score_mat',  'sub_mat',  'seqr',  'seqc',  'path',  'nCol',  'nRow',  'gap_penalty', 'l'])
#    for i in range(1,nRow):
#        for j in range(1,nCol):
#            rn = seqr[i-1]
#            cn = seqc[j-1]
#            if (rn>cn):
#                tmp = rn
#                rn = cn
#                cn = tmp
#            ind = (2*l - rn +1)*rn/2 + cn -rn
#            f = [0, 0, 0, 0]
#            f[1] = score_mat[ i-1 ][ j-1 ] + sub_mat[ ind ]
#            f[2] = score_mat[ i ][ j-1 ] + gap_penalty
#            f[3] = score_mat[ i-1 ][ j ] + gap_penalty
#            direction = [-1, ind,  -2, -3]
#            score_mat[i][j] = max(f)
#            path[i][j] = direction[ f.index( max(f) ) ]
    
    #highest score
    score = np.amax( score_mat )
    
    # trace back
    if show_alignment==1:
        pos = np.argmax( score_mat )
        row,  col = pos/nCol,  pos% nCol 
        alignRow = ''
        alignCol = ''
        p = path[ row ][ col ]
        while p!=-1:
            if p==-2:
                alignRow = '-' + alignRow
                alignCol = seqCol[ col-1 ] + alignCol
                col = col-1
            elif p==-3:
                alignRow = seqRow[ row -1] + alignRow
                alignCol = '-'+ alignCol
                row = row-1
            else: 
                alignRow = seqRow[ row-1 ] + alignRow
                alignCol = seqCol[ col-1 ] + alignCol
                row,  col = row-1,  col-1
            p = path[ row ][ col ]
        print alignRow
        print alignCol
        return score
    elif show_alignment == 2:
        pos = int(np.argmax( score_mat ) )
        path_code = np.zeros(len(para_ind))
        row,  col = pos/nCol,  pos% nCol 
        p = int(path[ row ][ col ])
        while p!=-1:
            if p==-2:
                if gap_ind in para_dic:
                    path_code[para_dic[gap_ind]] += 1
                col = col-1
            elif p==-3:
                if gap_ind in para_dic:
                    path_code[para_dic[gap_ind]] += 1
                row = row-1
            else:
                if p in para_dic:
                    path_code[ para_dic[p] ] += 1 
                row,  col = row-1,  col-1
            p = int(path[ row ][ col ])
        return score, path_code
    else:
        return score
