from itertools import combinations, product
from collections import defaultdict

import numpy as np
from scipy.optimize import milp, LinearConstraint


def allocate(
    data,
    linkvar=None,
    ngroups=4,
    must_allocate_all=True,
    max_size_difference=np.inf,
    balance_histogram={},
    balance_mean={},
    must_be_in_group={},
    cant_be_in_group={},
    group_objective={},
    max_unique_linked={},
    balance_pairwise=False,
):
    nt = len(data)
    c = -np.ones((ngroups, nt))
    for g, objective in group_objective.items():
        c[g] = objective
    c = c.flatten()

    tarr = np.array([list(range(nt))] * ngroups).flatten()
    garr = np.array([list(range(ngroups))] * nt).T.flatten()

    ####################################
    ### CONSTRAINTS

    # each datapoint can only be in one group
    tcA = np.zeros((nt, nt * ngroups))
    for t in range(nt):
        tcA[t, tarr == t] = 1
    tcub = np.ones(nt)
    # if must_allocate_all, each datapoint must be in a group
    tclb = np.ones(nt) if must_allocate_all else np.zeros(nt)

    # # each group should have at least n datapoints
    # n = 45
    # gcA = np.zeros((ngroups, nt * ngroups))
    # for g in range(ngroups):
    #     gcA[g, garr == g] = 1
    # gcub = np.ones(ngroups) * np.inf
    # gclb = np.ones(ngroups) * n

    # difference between number of datapoints in each group should be no more than nd
    gdiffAlist = []
    # difference between number at each level of each var in balance_histogram should
    # be no more than set number
    gdiffhistAlist = defaultdict(list)
    gdiffhistlb = {}
    gdiffhistub = {}
    catvecs = {}
    for var in balance_histogram:
        catvecs[var] = [
            np.array([np.array((data[var] == val))] * ngroups).flatten()
            for val in data[var].dropna().unique()
        ]
    # difference between mean of each var in balance_mean should be no more than
    # set number
    gdiffmeanAlist = defaultdict(list)
    gdiffmeanlb = {}
    gdiffmeanub = {}
    valvecs = {}
    for var in balance_mean:
        valvecs[var] = np.array([np.array(data[var])] * ngroups).flatten()
    if balance_pairwise:
        giter = combinations(range(ngroups), 2)
    else:
        giter = range(ngroups)
    for gval in giter:
        if balance_pairwise:
            g1, g2 = gval
            pairdiffA = (garr == g1).astype(int) - (garr == g2).astype(int)
        else:
            g = gval
            pairdiffA = (garr == g).astype(float) - (garr != g).astype(float) / (
                ngroups - 1
            )
        ####
        for var in balance_histogram:
            pairdiffvarAlist = []
            for cv in catvecs[var]:
                if balance_pairwise:
                    pairdiffvarA_ = ((garr == g1) & cv).astype(int) - (
                        (garr == g2) & cv
                    ).astype(int)
                else:
                    pairdiffvarA_ = ((garr == g) & cv).astype(float) - (
                        (garr != g) & cv
                    ).astype(float) / (ngroups - 1)
                pairdiffvarAlist.append(pairdiffvarA_[None])
            pairdiffvarA = np.concatenate(pairdiffvarAlist, axis=0)
            gdiffhistAlist[var].append(pairdiffvarA)
        ####
        for var in balance_mean:
            gdiffmeanAlist[var].append(
                ((pairdiffA / (nt / ngroups)) * valvecs[var])[None]
            )
        gdiffAlist.append(pairdiffA[None])

    gdiffA = np.concatenate(gdiffAlist, axis=0)
    gdifflb = np.ones(gdiffA.shape[0]) * -max_size_difference
    gdiffub = np.ones(gdiffA.shape[0]) * max_size_difference
    gdiffhistA = {}
    gdiffmeanA = {}
    for var, n in balance_histogram.items():
        gdiffhistA[var] = np.concatenate(gdiffhistAlist[var], axis=0)
        gdiffhistlb[var] = np.ones(gdiffhistA[var].shape[0]) * -n
        gdiffhistub[var] = np.ones(gdiffhistA[var].shape[0]) * n
    for var, n in balance_mean.items():
        gdiffmeanA[var] = np.concatenate(gdiffmeanAlist[var], axis=0)
        gdiffmeanlb[var] = np.ones(gdiffmeanA[var].shape[0]) * -n
        gdiffmeanub[var] = np.ones(gdiffmeanA[var].shape[0]) * n

    # each pair of linked datapoints, indicated by linkvar,
    # should be in the same group
    if linkvar is not None:
        sgAlist = []
        for _, linked_data in data.groupby(linkvar):
            schlA = []
            for comb in combinations(linked_data.index, 2):
                if not comb:
                    continue
                # print(comb)
                combA = []
                for g in range(ngroups):
                    combgA = ((tarr == comb[0]) & (garr == g)).astype(int) - (
                        (tarr == comb[1]) & (garr == g)
                    ).astype(int)
                    combA.append(combgA[None])
                schlA.append(np.concatenate(combA, axis=0))
            if schlA:
                sgAlist.append(np.concatenate(schlA, axis=0))
        sgA = np.concatenate(sgAlist, axis=0)
        sglb = np.zeros(sgA.shape[0])
        sgub = np.zeros(sgA.shape[0])
    else:
        sgA = np.zeros((1, ngroups * nt))
        sglb = np.zeros(1)
        sgub = np.zeros(1)

    # No more than the set amount of unique linked items for a given group
    if max_unique_linked:
        linkvals = data[linkvar]
        ulinkvals = linkvals.unique()
        nmul = len(max_unique_linked)
        mulA = np.zeros((nmul, nt * ngroups))
        mullb = np.zeros(nmul)
        mulub = np.zeros(nmul)
        for i, (g, mul) in enumerate(max_unique_linked.items()):
            for ulv in ulinkvals:
                lvfilt = np.array([np.array(linkvals == ulv)] * ngroups).flatten()
                gfilt = garr == g
                filt = lvfilt & gfilt
                filtvals = (filt / filt.sum())[filt]
                mulA[i, filt] = filtvals
            mulub[i] = mul
    else:
        mulA = np.zeros((1, nt * ngroups))
        mullb = np.zeros(1)
        mulub = np.zeros(1)

    # these datapoints must be in a specific group
    mbAlist = []
    for t, mbg in must_be_in_group.items():
        mbA_ = np.zeros(ngroups * nt)
        for g in mbg:
            mbA_[(tarr == t) & (garr == g)] = 1
        mbAlist.append(mbA_[None])
    if mbAlist:
        mbA = np.concatenate(mbAlist, axis=0)
        mblb = np.ones(mbA.shape[0])
        mbub = np.ones(mbA.shape[0])
    else:
        mbA = np.zeros((1, ngroups * nt))
        mblb = np.zeros(mbA.shape[0])
        mbub = np.zeros(mbA.shape[0])

    # these datapoints can't be in a specific group
    cbAlist = []
    for t, cbg in cant_be_in_group.items():
        cbA_ = np.zeros(ngroups * nt)
        for g in cbg:
            cbA_[(tarr == t) & (garr == g)] = 1
        cbAlist.append(cbA_[None])
    if cbAlist:
        cbA = np.concatenate(cbAlist, axis=0)
        cblb = np.zeros(cbA.shape[0])
        cbub = np.zeros(cbA.shape[0])
    else:
        cbA = np.zeros((1, ngroups * nt))
        cblb = np.zeros(cbA.shape[0])
        cbub = np.zeros(cbA.shape[0])

    Ab = [tcA, gdiffA, *gdiffhistA.values(), *gdiffmeanA.values(), sgA, mbA, cbA, mulA]
    lb = [
        tclb,
        gdifflb,
        *gdiffhistlb.values(),
        *gdiffmeanlb.values(),
        sglb,
        mblb,
        cblb,
        mullb,
    ]
    ub = [
        tcub,
        gdiffub,
        *gdiffhistub.values(),
        *gdiffmeanub.values(),
        sgub,
        mbub,
        cbub,
        mulub,
    ]

    const = LinearConstraint(
        A=np.concatenate(Ab, axis=0),
        ub=np.concatenate(ub, axis=0),
        lb=np.concatenate(lb, axis=0),
    )

    ####################################

    tol = 1e-6
    res = milp(c, integrality=np.ones_like(c) * 3, constraints=const)
    if res.success:
        x = (res.x > tol).astype(int)
        return x.reshape((ngroups, nt))
    else:
        print(res)
        return
