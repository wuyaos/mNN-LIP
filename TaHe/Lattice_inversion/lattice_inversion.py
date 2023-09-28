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

def generate_semigroup(list_a: np.ndarray) -> np.ndarray:
    list_a.sort()
    N = list_a.size
    while True:
        list_b = np.unique(np.kron(list_a, list_a))
        list_b.sort()
        list_c = list_b[:N]
        if np.allclose(list_a, list_c, rtol=1e-4, atol=1e-4):
            return list_c
        else:
            list_a = list_c.copy()


def add(a, b):
    # 将 a 重复 b 的行数次
    repeated_a = np.repeat(a, b.shape[0], axis=0)
    # 将 b 扩展为与 repeated_a 相同的形状
    tiled_b = np.tile(b, (a.shape[0], 1))
    # 将重复后的 a 和扩展后的 b 相加
    c = repeated_a + tiled_b
    return c


def store2pdict(site, pdict):
    # 每一行平方求和
    b0n2 = np.sum(np.power(site, 2), axis=1)
    # b0n2的每个元素都是一个字典，字典的key是b0n2，value是对应的个数
    k, n = np.unique(b0n2, return_counts=True)
    # 将字典pdict的key和value分别存入k和n, key相同则相加
    for i in range(len(k)):
        pdict[k[i]] = pdict.get(k[i], 0) + n[i]


def get_shifts(LATTYPE: str = "sc"):
    if LATTYPE.lower() == "sc":
        shift = np.array([[0, 0, 0]])
    elif LATTYPE.lower() == "bcc":
        shift = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    elif LATTYPE.lower() == "fcc":
        shift = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    elif LATTYPE.lower() == "dia":
        shift = np.array([[0.00, 0.00, 0.00], [0.50, 0.50, 0.00], [0.00, 0.50, 0.50], [0.50, 0.00, 0.50], [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]])
    elif LATTYPE.lower() == "b3":
        shift = np.array([[0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]])
    elif LATTYPE.lower() == "b1":
        shift = np.array([[0.00, 0.00, 0.50], [0.00, 0.50, 0.00], [0.50, 0.00, 0.00], [0.50, 0.50, 0.50]])
    elif LATTYPE.lower() == "oct":
        shift = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]])
    elif LATTYPE.lower() == "tet":
        shift = np.array([[0.5, 0.75, 0.0], [0.0, 0.25, 0.5]])
    else:
        raise ValueError("LATTYPE must be one of sc, bcc, fcc, dia, b3, b1")
    return shift


def generate_latinvpara(shift: np.ndarray, NITEM: int = 20):
    pdict = {}  # key: b0(n)^2 => value: r0(n)
    LIMIT = 10
    i, j, k = np.meshgrid(np.arange(-LIMIT, LIMIT + 1), np.arange(-LIMIT, LIMIT + 1), np.arange(-LIMIT, LIMIT + 1), indexing='ij')
    site = np.stack((i.flatten(), j.flatten(), k.flatten()), axis=1)
    site = add(site, shift)
    store2pdict(site, pdict)

    # 删除 0 元素
    if 0.0 in pdict:
        pdict.pop(0.0, None)

    # dict 转换为两个排序数组:b0n2Arr r0nArr
    b0n2Arr = sorted(pdict)
    r0nArr = np.array([pdict[b0n2] for b0n2 in b0n2Arr])

    # 使b0n2 一个元素等于 1.0
    b0n2Arr /= b0n2Arr[0]

    # 1.获得NITEM个元素
    b0n2Arr = b0n2Arr[0:NITEM]
    r0nArr = r0nArr[0:NITEM]
    b0nArr = np.sqrt(b0n2Arr)

    # 2:扩展b0(n)^2 => b(n)^2
    bn2Arr = generate_semigroup(b0n2Arr)
    bnArr = np.sqrt(bn2Arr)

    # 3:计算 r (n)
    rnArr = np.zeros(NITEM)
    _, idx1, idx2 = np.intersect1d(b0n2Arr, bn2Arr, return_indices=True)
    rnArr[idx2] = r0nArr[idx1]

    # 4:计算I(n)
    InArr = np.zeros(NITEM)
    InArr[0] = 1.0 / rnArr[0]
    for k in range(1, NITEM):
        fraction = bn2Arr[k] / bn2Arr[:k]
        idx = np.argwhere(np.abs(bn2Arr - fraction.reshape(-1, 1)) < 1e-10)
        sum_In = np.sum(InArr[idx[:, 0]] * rnArr[idx[:, 1]])
        InArr[k] = -sum_In / rnArr[0]
    return bnArr, rnArr, InArr


def get_r0b0(shift: np.ndarray):
    pdict = {}  # key: b0(n)^2 => value: r0(n)
    LIMIT = 10
    i, j, k = np.meshgrid(np.arange(-LIMIT, LIMIT + 1), np.arange(-LIMIT, LIMIT + 1), np.arange(-LIMIT, LIMIT + 1), indexing='ij')
    site = np.stack((i.flatten(), j.flatten(), k.flatten()), axis=1)
    site = add(site, shift)
    store2pdict(site, pdict)

    # 删除 0 元素
    if 0.0 in pdict:
        pdict.pop(0.0, None)

    # dict 转换为两个排序数组:b0n2Arr r0nArr
    b0n2Arr = sorted(pdict)
    r0nArr = np.array([pdict[b0n2] for b0n2 in b0n2Arr])
    return b0n2Arr, r0nArr


def generate_latinvpara_mix(shift1: np.ndarray, shift2: np.ndarray, NITEM: int = 20):
    b0n2Arr1, r0nArr1 = get_r0b0(shift1)
    b0n2Arr2, r0nArr2 = get_r0b0(shift2)

    # 返回共同元素的索引以及各自非共同元素的索引
    _, idx1, idx2 = np.intersect1d(b0n2Arr1, b0n2Arr2, return_indices=True)
    if idx1.size > 0:
        idx2_ = np.setdiff1d(np.arange(len(b0n2Arr2)), idx2)
        b0n2Arr = np.concatenate((b0n2Arr1, b0n2Arr2[idx2_]))
        r0nArr = np.concatenate((r0nArr1, -r0nArr2[idx2_]))
        r0nArr[idx1] = r0nArr1[idx1] - r0nArr2[idx2]
    else:
        b0n2Arr = np.concatenate((b0n2Arr1, b0n2Arr2))
        r0nArr = np.concatenate((r0nArr1, -r0nArr2))

    # 按照b0n2Arr排序
    idx = np.argsort(b0n2Arr)
    b0n2Arr = b0n2Arr[idx]
    r0nArr = r0nArr[idx]

    # 第一个元素等于1
    b02nArr = b0n2Arr / b0n2Arr[0]

    # NITEM
    b02nArr = b02nArr[0:NITEM]
    r0nArr = r0nArr[0:NITEM]
    b0nArr = np.sqrt(b02nArr)

    # b0 => bn
    bn2Arr = generate_semigroup(b02nArr)
    bnArr = np.sqrt(bn2Arr)

    # rn
    rnArr = np.zeros(NITEM)
    _, idx1, idx2 = np.intersect1d(b02nArr, bn2Arr, return_indices=True)
    rnArr[idx2] = r0nArr[idx1]

    # In
    InArr = np.zeros(NITEM)
    InArr[0] = 1.0 / rnArr[0]
    for k in range(1, NITEM):
        fraction = bn2Arr[k] / bn2Arr[:k]
        idx = np.argwhere(np.abs(bn2Arr - fraction.reshape(-1, 1)) < 1e-10)
        sum_In = np.sum(InArr[idx[:, 0]] * rnArr[idx[:, 1]])
        InArr[k] = -sum_In / rnArr[0]

    return bnArr, rnArr, InArr

# 输出Sc结构的晶格反演参数
bnArr, rnArr, InArr = generate_latinvpara_mix(get_shifts("oct"), get_shifts("tet"), NITEM=100)
import pandas as pd
df = pd.DataFrame({"bn": bnArr, "rn": rnArr, "In": InArr})
df.to_csv("Oct_latinvpara.csv", index=False)
