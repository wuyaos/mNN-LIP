# -*- encoding: utf-8 -*-
import numpy as np

def zblSec(r,z1,z2):
    '''
    Usage: calculate ZBL potential.
    Input:
        r: atomic distance. unit: A
        z1: atomic number of 1st element
        z2: atomic number of 2nd element
    Return:
        frij: zbl potential w.r.t. r
        dfrij: force w.r.t. r
    '''
    frij = dfrij = 0.0

    c = [0.18175, 0.50986, 0.28022, 0.02817]
    d = [3.19980, 0.94229, 0.40290, 0.20162]
    a0 = 0.46850
    cc = 14.3997
    tt = cc*z1*z2

    a = a0/(z1**0.23+z2**0.23)
    x = r/a

    phi = c[0]*np.exp(-d[0]*x) + c[1]*np.exp(-d[1]*x) + c[2]*np.exp(-d[2]*x) + c[3]*np.exp(-d[3]*x)
    dphi= c[0]*np.exp(-d[0]*x)*(-d[0]) + c[1]*np.exp(-d[1]*x)*(-d[1]) + c[2]*np.exp(-d[2]*x)*(-d[2]) + c[3]*np.exp(-d[3]*x)*(-d[3])

    frij = tt*phi/r
    dfrij = (-tt/(r*r))*phi+(tt/r)*(dphi/a)

    return frij

def dzblSec(r,z1,z2):
    '''
    Usage: calculate ZBL potential.
    Input:
        r: atomic distance. unit: A
        z1: atomic number of 1st element
        z2: atomic number of 2nd element
    Return:
        frij: zbl potential w.r.t. r
        dfrij: force w.r.t. r
    '''
    frij = dfrij = 0.0

    c = [0.18175, 0.50986, 0.28022, 0.02817]
    d = [3.19980, 0.94229, 0.40290, 0.20162]
    a0 = 0.46850
    cc = 14.3997
    tt = cc*z1*z2

    a = a0/(z1**0.23+z2**0.23)
    x = r/a

    phi = c[0]*np.exp(-d[0]*x) + c[1]*np.exp(-d[1]*x) + c[2]*np.exp(-d[2]*x) + c[3]*np.exp(-d[3]*x)
    dphi= c[0]*np.exp(-d[0]*x)*(-d[0]) + c[1]*np.exp(-d[1]*x)*(-d[1]) + c[2]*np.exp(-d[2]*x)*(-d[2]) + c[3]*np.exp(-d[3]*x)*(-d[3])

    frij = tt*phi/r
    dfrij = (-tt/(r*r))*phi+(tt/r)*(dphi/a)

    return dfrij



# 半群乘法
def generate_semigroup(list_in):
    list_in_sorted = np.sort(list_in, axis=0)
    N = list_in_sorted.size
    list_tmp1 = list_in_sorted.copy()
    while True:
        list_tmp2 = list_tmp1.copy()
        for i in range(N):
            for j in range(N):
                list_tmp2 = np.append(list_tmp2, list_tmp2[i] * list_tmp2[j])
        list_tmp2 = np.unique(list_tmp2)[0:N]
        if np.allclose(list_tmp1, list_tmp2, rtol=1e-04, atol=1e-04, equal_nan=False):
            break
        else:
            list_tmp1 = list_tmp2[0:N].copy()
    return list_tmp1


def store2pdict(i, j, k, pdict):
    b0n2 = np.power(i, 2) + np.power(j, 2) + np.power(k, 2)
    if b0n2 in pdict:
        pdict[b0n2] = pdict[b0n2] + 1
    else:
        pdict[b0n2] = 1


def generate_latinvpara(LATTYPE, NITEM=20):

    b0n2Arr = []
    r0nArr = []
    rnArr = []
    InArr = []

    # Step 1: calculate  b0(n)^2  and  r0(n)

    pdict = {}  # key: b0(n)^2 => value: r0(n)
    LIMIT = 10
    for i in range(-LIMIT, LIMIT):
        for j in range(-LIMIT, LIMIT):
            for k in range(-LIMIT, LIMIT):
                if LATTYPE.lower() == "sc":  # for SC
                    store2pdict(i, j, k, pdict)
                elif LATTYPE.lower() == "bcc":  # for BCC
                    store2pdict(i, j, k, pdict)
                    store2pdict(i + 0.5, j + 0.5, k + 0.5, pdict)
                elif LATTYPE.lower() == "fcc":  # for FCC
                    store2pdict(i, j, k, pdict)
                    store2pdict(i + 0.5, j + 0.5, k, pdict)
                    store2pdict(i + 0.5, j, k + 0.5, pdict)
                    store2pdict(i, j + 0.5, k + 0.5, pdict)
                elif LATTYPE.lower() == "dia":  # for dia
                    store2pdict(i, j, k, pdict)
                    store2pdict(i + 0.5, j + 0.5, k, pdict)
                    store2pdict(i + 0.5, j, k + 0.5, pdict)
                    store2pdict(i, j + 0.5, k + 0.5, pdict)
                    store2pdict(i + 0.25, j + 0.25, k + 0.25, pdict)
                    store2pdict(i + 0.75, j + 0.75, k + 0.25, pdict)
                    store2pdict(i + 0.75, j + 0.25, k + 0.75, pdict)
                    store2pdict(i + 0.25, j + 0.75, k + 0.75, pdict)
                elif LATTYPE.lower() == "b3":  # for dia
                    store2pdict(i + 0.25, j + 0.25, k + 0.25, pdict)
                    store2pdict(i + 0.25, j + 0.75, k + 0.75, pdict)
                    store2pdict(i + 0.75, j + 0.25, k + 0.75, pdict)
                    store2pdict(i + 0.75, j + 0.75, k + 0.25, pdict)
                elif LATTYPE.lower() == "b1":  # for dia
                    store2pdict(i, j, k + 0.5, pdict)
                    store2pdict(i, j + 0.5, k, pdict)
                    store2pdict(i + 0.5, j, k, pdict)
                    store2pdict(i + 0.5, j + 0.5, k + 0.5, pdict)
                else:
                    print(LATTYPE, " is NOT supported now! Supported: SC/BCC/FCC/DIA")
                    return

    # remove the 0 element
    if 0.0 in pdict:
        pdict.pop(0.0)

    # convert dict to two sorted lists: b0n2Arr and r0nArr

    for b0n2 in sorted(pdict):
        b0n2Arr.append(b0n2)
        r0nArr.append(pdict[b0n2])

    # normalize b0n2 so that the first element equals 1.0
    b0n2Arr = [tmp / b0n2Arr[0] for tmp in b0n2Arr]

    # obtain the leading NITEM
    b0n2Arr = b0n2Arr[0:NITEM]
    r0nArr = r0nArr[0:NITEM]
    b0nArr = [np.sqrt(i) for i in b0n2Arr]

    # Step 2: extend b0(n)^2 to b(n)^2
    bn2Arr = generate_semigroup(b0n2Arr)

    bnArr = [np.sqrt(i) for i in bn2Arr]

    # Step 3: calculate r(n)
    j = 0
    for i in range(NITEM):
        if bn2Arr[i] in b0n2Arr:
            rnArr.append(r0nArr[j])
            j = j + 1
        else:
            rnArr.append(0)

    # Step 4: calculate I(n)
    InArr.append(1.0 / rnArr[0])
    for i in range(1, NITEM):
        k = i
        sum_In = 0.0
        for j in range(k):
            for m in range(NITEM):
                if abs(bn2Arr[k] / bn2Arr[j] - bn2Arr[m]) < 1e-10:
                    sum_In = sum_In + InArr[j] * rnArr[m]
        InArr.append(-sum_In / rnArr[0])
    return bnArr, InArr, rnArr

def generate_latinvpara2(NITEM=20):

    b1_b0n2Arr = []
    b1_r0nArr = []
    b1_rnArr = []
    b1_InArr = []
    b3_b0n2Arr = []
    b3_r0nArr = []
    b3_rnArr = []
    b3_InArr = []

    # Step 1: calculate  b0(n)^2  and  r0(n)

    b1_pdict = {}  # key: b0(n)^2 => value: r0(n)
    b3_pdict = {}  # key: b0(n)^2 => value: r0(n)
    LIMIT = 10
    # B3
    for i in range(-LIMIT, LIMIT):
        for j in range(-LIMIT, LIMIT):
            for k in range(-LIMIT, LIMIT):
                store2pdict(i + 0.25, j + 0.25, k + 0.25, b3_pdict)
                store2pdict(i + 0.25, j + 0.75, k + 0.75, b3_pdict)
                store2pdict(i + 0.75, j + 0.25, k + 0.75, b3_pdict)
                store2pdict(i + 0.75, j + 0.75, k + 0.25, b3_pdict)
    # B1
    for i in range(-LIMIT, LIMIT):
        for j in range(-LIMIT, LIMIT):
            for k in range(-LIMIT, LIMIT):
                store2pdict(i, j, k + 0.5, b1_pdict)
                store2pdict(i, j + 0.5, k, b1_pdict)
                store2pdict(i + 0.5, j, k, b1_pdict)
                store2pdict(i + 0.5, j + 0.5, k + 0.5, b1_pdict)

    # remove the 0 element
    if 0.0 in b1_pdict:
        b1_pdict.pop(0.0)

    # remove the 0 element
    if 0.0 in b3_pdict:
        b3_pdict.pop(0.0)

    # convert dict to two sorted lists: b0n2Arr and r0nArr

    for b1_b0n2 in sorted(b1_pdict):
        b1_b0n2Arr.append(b1_b0n2)
        b1_r0nArr.append(b1_pdict[b1_b0n2])

    for b3_b0n2 in sorted(b3_pdict):
        b3_b0n2Arr.append(b3_b0n2)
        b3_r0nArr.append(b3_pdict[b3_b0n2])

    # # normalize b0n2 so that the first element equals 1.0
    # b1_b0n2Arr = [tmp / b1_b0n2Arr[0] for tmp in b1_b0n2Arr]
    # b3_b0n2Arr = [tmp / b3_b0n2Arr[0] for tmp in b3_b0n2Arr]

    r0nArr = []
    index_b1 = []
    index_b3 = []
    for ind, i in enumerate(b3_b0n2Arr):
        delta = np.abs(b1_b0n2Arr - i)
        if np.where(delta <= 0.00001)[0].size > 0:
            index_b1.append(np.where(delta <= 0.00001)[0][0])
            index_b3.append(ind)

    b0n2Arr = list(b1_b0n2Arr) + list(np.delete(b3_b0n2Arr, index_b3))
    for ind, i in enumerate(b1_b0n2Arr):
        if ind in index_b1:
            b3_ind = index_b3[index_b1.index(ind)]
            r0nArr.append(b1_r0nArr[ind] - b3_r0nArr[b3_ind])
        else:
            r0nArr.append(b1_r0nArr[ind])
    for ind, i in enumerate(b3_b0n2Arr):
        if ind in index_b3:
            continue
        r0nArr.append(-b3_r0nArr[ind])

    full_arr = np.vstack((b0n2Arr, r0nArr))
    full_arr = full_arr.T[np.argsort(full_arr.T[:, 0])].T
    b0n2Arr = full_arr[0, :]
    r0nArr = full_arr[1, :]

    b0n2Arr = [tmp / b0n2Arr[0] for tmp in b0n2Arr]

    # obtain the leading NITEM
    b0n2Arr = b0n2Arr[0:NITEM]
    r0nArr = r0nArr[0:NITEM]
    b0nArr = [np.sqrt(i) for i in b0n2Arr]

    # Step 2: extend b0(n)^2 to b(n)^2
    bn2Arr = generate_semigroup(b0n2Arr)
    bnArr = [np.sqrt(i) for i in bn2Arr]

    rnArr = []
    InArr = []
    # Step 3: calculate r(n)
    j = 0
    for i in range(NITEM):
        if bn2Arr[i] in b0n2Arr:
            rnArr.append(r0nArr[j])
            j = j + 1
        else:
            rnArr.append(0)

    # Step 4: calculate I(n)
    InArr.append(1.0 / rnArr[0])
    for i in range(1, NITEM):
        k = i
        sum_In = 0.0
        for j in range(k):
            for m in range(NITEM):
                if abs(bn2Arr[k] / bn2Arr[j] - bn2Arr[m]) < 1e-10:
                    sum_In = sum_In + InArr[j] * rnArr[m]
        InArr.append(-sum_In / rnArr[0])
    return bnArr, InArr

