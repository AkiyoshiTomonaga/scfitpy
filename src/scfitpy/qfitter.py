"""qfitter.py -"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import h, physical_constants
from tqdm import tqdm

phi0 = physical_constants["mag. flux quantum"][0]

try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp


def band(n):
    """band matrix: M(i,j)=kron(i,j-1)"""
    band = np.array(np.zeros((n, n), dtype=np.complex64))
    for i in range(n - 1):
        band[i][i + 1] = 1
    return band


def dest(n):
    """annihilation operator: M(i,j)=sqrt(i)*kron(i,j-1)"""
    dest = np.array(np.zeros((n, n), dtype=np.complex64))
    for i in range(n - 1):
        dest[i][i + 1] = np.sqrt(i + 1)
    return dest


def nsq(n):
    nsq = np.array(np.zeros((2 * n + 1, 2 * n + 1), dtype=np.complex64))
    for i in range(2 * n + 1):
        nsq[i][i] = (i - n) ** 2
    return nsq


def nmat(n):
    nmat = np.array(np.zeros((2 * n + 1, 2 * n + 1), dtype=np.complex64))
    for i in range(2 * n + 1):
        nmat[i][i] = i - n
    return nmat


def phi1_mat(nmax):
    pb = np.array(np.zeros((2 * nmax + 1, 2 * nmax + 1), dtype=np.complex64))
    for m1 in range(-nmax, nmax + 1):
        for m2 in range(-nmax, nmax + 1):
            if m1 != m2:
                pb[m1 + nmax][m2 + nmax] = (1j * (-1) ** (m1 - m2)) / (m1 - m2)
    return pb


def phi2_mat(nmax):
    pb2 = np.array(np.zeros((2 * nmax + 1, 2 * nmax + 1), dtype=np.complex64))
    for m1 in range(-nmax, nmax + 1):
        for m2 in range(-nmax, nmax + 1):
            if m1 != m2:
                pb2[m1 + nmax][m2 + nmax] = (2 * (-1) ** (m1 - m2)) / (m1 - m2) ** 2
            else:
                pb2[m1 + nmax][m2 + nmax] = np.pi**2 / 3
    return pb2


sigz = [[1, 0], [0, -1]]
sigx = [[0, 1], [1, 0]]


def QHami(
    N: int, eps: float, g: float, d: float, omegaC0: float, L: float, R: float
) -> np.ndarray:
    """Get a Hamiltonian of ...

    Args:
        N: the number of dimension for the cavity (size of fock space)
        eps: xxx

    Return:
        Hamiltonian - np.ndarray[complex], dims=[N,N]
    """
    sz = np.kron(sigz, np.eye(N))
    sx = np.kron(sigx, np.eye(N))
    a = np.kron(np.eye(2), dest(N))
    if eps > 0:
        omega = omegaC0 * (1 + R * eps)
    else:
        omega = omegaC0 * (1 - L * eps)
    H = (
        eps * sz / 2
        + d * sx / 2
        + g * np.matmul(sz, a.T + a)
        + omega * np.matmul(a.T, a)
    )
    return H


def Rabi(flist, params):
    """Rabi-model Hamiltonian for fitting"""
    F = 13
    bn = 10
    E1 = [[] for i in range(bn)]
    for f in flist:
        H = QHami(
            F, f * params[5], params[0], params[1], params[2], params[3], params[4]
        )
        evals, ekets = np.linalg.eigh(H)
        for q in range(bn):
            E1[q].append(evals[q])
    return E1


def Rabi05(flist, params, df):
    F = 13
    bn = 10
    E1 = [[] for i in range(bn)]
    for f in flist:
        H = QHami(
            F,
            2 * (f - 0.5) * params[5] * df,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
        )
        evals, ekets = np.linalg.eigh(H)
        for q in range(bn):
            E1[q].append(evals[q])
    return E1


def circuit_spectrum(phi_list, params, nlist):
    Ej = params[0]
    Ec = params[1]
    Lr = params[2]
    Cr = params[3]
    alpha = params[4]
    beta = params[5]
    b = 1.0

    wr = 1e-9 / (2 * np.pi * np.sqrt(Cr * Lr))
    Ir = np.sqrt(h / (4 * np.pi * Lr * np.sqrt(Lr * Cr)))

    bn = 10  # haow many band will be shown
    E = [[] for i in range(bn)]
    E1 = [[] for i in range(bn)]
    nmax1 = nlist[0]
    nmax2 = nlist[1]
    nmax3 = nlist[2]
    fock = 15
    dim1 = 2 * nmax1 + 1
    dim2 = 2 * nmax2 + 1
    dim3 = 2 * nmax3 + 1

    # Matrix preparation
    eye1 = np.eye(dim1)
    eye2 = np.eye(dim2)
    eye3 = np.eye(dim3)
    eyef = np.eye(fock)

    # Hamiltonian construction
    # Kinetic energy part
    Nbt = np.kron(np.kron(nsq(nmax1), eye2), eye3)
    Nal = np.kron(np.kron(eye1, nsq(nmax2)), eye3)
    Nb = np.kron(eye1, np.kron(eye2, nsq(nmax3)))
    NbtNal = np.kron(np.kron(nmat(nmax1), nmat(nmax2)), eye3)
    NalNb = np.kron(np.kron(eye1, nmat(nmax2)), nmat(nmax3))
    NbtNb = np.kron(nmat(nmax1), np.kron(eye2, nmat(nmax3)))

    # Resonator
    Hr0 = wr * np.matmul(dest(fock).T, dest(fock))

    # phi and phi**2 term
    Pb = np.kron(phi1_mat(nmax1), np.kron(eye2, eye3))
    Pb2 = np.kron(phi2_mat(nmax1), np.kron(eye2, eye3))
    Hcr = dest(fock).T + dest(fock)
    coef = -Ir * (phi0 / (2 * np.pi)) / (h * 1e9)

    # Flux independent potential
    HL = (0.5 / (Lr * h * 1e9)) * (phi0 / (2 * np.pi)) ** 2 * Pb2

    band1 = np.kron(np.kron(band(dim1) + band(dim1).T, eye2), eye3)
    band2 = np.kron(np.kron(eye1, band(dim2) + band(dim2).T), eye3)
    band3 = np.kron(np.kron(eye1, eye2), band(dim3) + band(dim3).T)
    Hj = -0.5 * Ej * (beta * band1 + alpha * band2 + band3)
    bandf = np.kron(np.kron(band(dim1), band(dim2)), band(dim3))

    D = alpha * b * beta + alpha * b + alpha * beta + b * beta
    Cb = alpha * b + alpha + b
    C1 = b * beta + b + beta
    C2 = alpha * b + alpha * beta + b * beta
    Ht = (4 * Ec / D) * (
        Cb * Nbt
        + C1 * Nal
        + C2 * Nb
        - 2 * b * NbtNal
        - 2 * b * beta * NalNb
        - 2 * alpha * beta * NbtNb
    )

    for phi in tqdm(phi_list):  # Flux dependent part
        Hf = (
            -0.5
            * b
            * Ej
            * (bandf * np.exp(2j * np.pi * phi) + bandf.T * np.exp(-2j * np.pi * phi))
        )

        Hq0 = Ht + HL + Hj + Hf
        evals0, ekets0 = np.linalg.eigh(Hq0)

        for q in range(bn):
            E[q].append(evals0[q])
        qspace = 25
        psiq = []
        for k in range(qspace):
            psiq.append(ekets0[:, k])
        phib_mat = np.array(np.zeros((qspace, qspace), dtype=np.cdouble))
        hq0 = np.array(np.zeros((qspace, qspace), dtype=np.cdouble))
        for x in range(qspace):
            hq0[x][x] = evals0[x]
            for y in range(qspace):
                phib_mat[x][y] = np.dot(np.conjugate(psiq[x]), np.dot(Pb, psiq[y]))

        Hqco = np.kron(hq0, eyef)
        Hcco = coef * np.kron(phib_mat, Hcr)
        Hrco = np.kron(np.eye(qspace), Hr0)
        evals1, ekets1 = np.linalg.eigh(Hqco + Hcco + Hrco)

        for q in range(bn):
            E1[q].append(evals1[q])
    eps0 = np.array(E[1]) - np.array(E[0])
    eps = np.sign(phi_list - 0.5) * np.sqrt(abs(eps0**2 - min(eps0) ** 2))
    return E1, eps


def circuit_spectrum_gpu(phi_list, params, nlist):
    Ej = params[0]
    Ec = params[1]
    Lr = params[2]
    Cr = params[3]
    alpha = params[4]
    beta = params[5]
    b = 1.0

    wr = 1e-9 / (2 * np.pi * np.sqrt(Cr * Lr))
    Ir = np.sqrt(h / (4 * np.pi * Lr * np.sqrt(Lr * Cr)))

    bn = 10  # haow many band will be shown
    E = [[] for i in range(bn)]
    E1 = [[] for i in range(bn)]
    nmax1 = nlist[0]
    nmax2 = nlist[1]
    nmax3 = nlist[2]
    fock = 15
    dim1 = 2 * nmax1 + 1
    dim2 = 2 * nmax2 + 1
    dim3 = 2 * nmax3 + 1

    # Matrix preparation
    eye1 = np.eye(dim1)
    eye2 = np.eye(dim2)
    eye3 = np.eye(dim3)
    eyef = np.eye(fock)

    # Hamiltonian construction
    # Kinetic energy part
    Nbt = np.kron(np.kron(nsq(nmax1), eye2), eye3)
    Nal = np.kron(np.kron(eye1, nsq(nmax2)), eye3)
    Nb = np.kron(eye1, np.kron(eye2, nsq(nmax3)))
    NbtNal = np.kron(np.kron(nmat(nmax1), nmat(nmax2)), eye3)
    NalNb = np.kron(np.kron(eye1, nmat(nmax2)), nmat(nmax3))
    NbtNb = np.kron(nmat(nmax1), np.kron(eye2, nmat(nmax3)))

    # Resonator
    Hr0 = wr * np.matmul(dest(fock).T, dest(fock))

    # phi and phi**2 term
    Pb = np.kron(phi1_mat(nmax1), np.kron(eye2, eye3))
    Pb2 = np.kron(phi2_mat(nmax1), np.kron(eye2, eye3))
    Hcr = dest(fock).T + dest(fock)
    coef = -Ir * (phi0 / (2 * np.pi)) / (h * 1e9)

    # Flux independent potential
    HL = (0.5 / (Lr * h * 1e9)) * (phi0 / (2 * np.pi)) ** 2 * Pb2

    band1 = np.kron(np.kron(band(dim1) + band(dim1).T, eye2), eye3)
    band2 = np.kron(np.kron(eye1, band(dim2) + band(dim2).T), eye3)
    band3 = np.kron(np.kron(eye1, eye2), band(dim3) + band(dim3).T)
    Hj = -0.5 * Ej * (beta * band1 + alpha * band2 + band3)
    bandf = np.kron(np.kron(band(dim1), band(dim2)), band(dim3))

    D = alpha * b * beta + alpha * b + alpha * beta + b * beta
    Cb = alpha * b + alpha + b
    C1 = b * beta + b + beta
    C2 = alpha * b + alpha * beta + b * beta
    Ht = (4 * Ec / D) * (
        Cb * Nbt
        + C1 * Nal
        + C2 * Nb
        - 2 * b * NbtNal
        - 2 * b * beta * NalNb
        - 2 * alpha * beta * NbtNb
    )

    for phi in tqdm(phi_list):  # Flux dependent part
        Hf = (
            -0.5
            * b
            * Ej
            * (bandf * np.exp(2j * np.pi * phi) + bandf.T * np.exp(-2j * np.pi * phi))
        )

        Hq0 = cp.array(Ht + HL + Hj + Hf, dtype=cp.complex64)
        evals01, ekets01 = cp.linalg.eigh(Hq0)
        evals0 = evals01.get()
        ekets0 = ekets01.get()

        for q in range(bn):
            E[q].append(evals0[q])
        qspace = 25
        psiq = []
        for k in range(qspace):
            psiq.append(ekets0[:, k])
        phib_mat = np.array(np.zeros((qspace, qspace), dtype=np.cdouble))
        hq0 = np.array(np.zeros((qspace, qspace), dtype=np.cdouble))
        for x in range(qspace):
            hq0[x][x] = evals0[x]
            for y in range(qspace):
                phib_mat[x][y] = np.dot(np.conjugate(psiq[x]), np.dot(Pb, psiq[y]))

        Hqco = np.kron(hq0, eyef)
        Hcco = coef * np.kron(phib_mat, Hcr)
        Hrco = np.kron(np.eye(qspace), Hr0)
        H_final = cp.array(Hqco + Hcco + Hrco, dtype=cp.complex64)
        evals11, ekets11 = cp.linalg.eigh(H_final)
        evals1 = evals11.get()
        ekets1 = ekets11.get()

        for q in range(bn):
            E1[q].append(evals1[q])
    eps0 = np.array(E[1]) - np.array(E[0])
    eps = np.sign(phi_list - 0.5) * np.sqrt(abs(eps0**2 - min(eps0) ** 2))
    return E1, eps


def circuit_spectrum_Q(phi_list, params, nlist):
    Ej = params[0]
    Ec = params[1]
    Lr = params[2]
    Cr = params[3]
    alpha = params[4]
    beta = params[5]
    b = 1.0

    wr = 1e-9 / (2 * np.pi * np.sqrt(Cr * Lr))
    Ir = np.sqrt(h / (4 * np.pi * Lr * np.sqrt(Lr * Cr)))

    bn = 10  # haow many band will be shown
    E = [[] for i in range(bn)]
    E1 = [[] for i in range(bn)]
    nmax1 = nlist[0]
    nmax2 = nlist[1]
    nmax3 = nlist[2]
    fock = 15
    dim1 = 2 * nmax1 + 1
    dim2 = 2 * nmax2 + 1
    dim3 = 2 * nmax3 + 1

    # Matrix preparation
    eye1 = np.eye(dim1)
    eye2 = np.eye(dim2)
    eye3 = np.eye(dim3)
    eyef = np.eye(fock)

    # Hamiltonian construction
    # Kinetic energy part
    Nbt = np.kron(np.kron(nsq(nmax1), eye2), eye3)
    Nal = np.kron(np.kron(eye1, nsq(nmax2)), eye3)
    Nb = np.kron(eye1, np.kron(eye2, nsq(nmax3)))
    NbtNal = np.kron(np.kron(nmat(nmax1), nmat(nmax2)), eye3)
    NalNb = np.kron(np.kron(eye1, nmat(nmax2)), nmat(nmax3))
    NbtNb = np.kron(nmat(nmax1), np.kron(eye2, nmat(nmax3)))

    # Resonator
    Hr0 = wr * np.matmul(dest(fock).T, dest(fock))

    # phi and phi**2 term
    Pb = np.kron(phi1_mat(nmax1), np.kron(eye2, eye3))
    Pb2 = np.kron(phi2_mat(nmax1), np.kron(eye2, eye3))
    Hcr = dest(fock).T + dest(fock)
    coef = -Ir * (phi0 / (2 * np.pi)) / (h * 1e9)

    # Flux independent potential
    HL = (0.5 / (Lr * h * 1e9)) * (phi0 / (2 * np.pi)) ** 2 * Pb2

    band1 = np.kron(np.kron(band(dim1) + band(dim1).T, eye2), eye3)
    band2 = np.kron(np.kron(eye1, band(dim2) + band(dim2).T), eye3)
    band3 = np.kron(np.kron(eye1, eye2), band(dim3) + band(dim3).T)
    Hj = -0.5 * Ej * (beta * band1 + alpha * band2 + band3)
    bandf = np.kron(np.kron(band(dim1), band(dim2)), band(dim3))

    D = alpha * b * beta + alpha * b + alpha * beta + b * beta
    Cb = alpha * b + alpha + b
    C1 = b * beta + b + beta
    C2 = alpha * b + alpha * beta + b * beta
    Ht = (4 * Ec / D) * (
        Cb * Nbt
        + C1 * Nal
        + C2 * Nb
        - 2 * b * NbtNal
        - 2 * b * beta * NalNb
        - 2 * alpha * beta * NbtNb
    )

    for phi in tqdm(phi_list):  # Flux dependent part
        Hf = (
            -0.5
            * b
            * Ej
            * (bandf * np.exp(2j * np.pi * phi) + bandf.T * np.exp(-2j * np.pi * phi))
        )

        Hq0 = Ht + HL + Hj + Hf
        evals0, ekets0 = np.linalg.eigh(Hq0)

        for q in range(bn):
            E1[q].append(evals0[q])

    return np.array(E1)


def circuit_spectrum_Qket(phi_list, params, nlist):
    Ej = params[0]
    Ec = params[1]
    Lr = params[2]
    Cr = params[3]
    alpha = params[4]
    beta = params[5]
    b = 1.0

    nmax1 = nlist[0]
    nmax2 = nlist[1]
    nmax3 = nlist[2]
    dim1 = 2 * nmax1 + 1
    dim2 = 2 * nmax2 + 1
    dim3 = 2 * nmax3 + 1

    # Matrix preparation
    eye1 = np.eye(dim1)
    eye2 = np.eye(dim2)
    eye3 = np.eye(dim3)

    # Hamiltonian construction
    # Kinetic energy part
    Nbt = np.kron(np.kron(nsq(nmax1), eye2), eye3)
    Nal = np.kron(np.kron(eye1, nsq(nmax2)), eye3)
    Nb = np.kron(eye1, np.kron(eye2, nsq(nmax3)))
    NbtNal = np.kron(np.kron(nmat(nmax1), nmat(nmax2)), eye3)
    NalNb = np.kron(np.kron(eye1, nmat(nmax2)), nmat(nmax3))
    NbtNb = np.kron(nmat(nmax1), np.kron(eye2, nmat(nmax3)))

    # phi**2 term
    Pb2 = np.kron(phi2_mat(nmax1), np.kron(eye2, eye3))

    # Flux independent potential
    HL = (0.5 / (Lr * h * 1e9)) * (phi0 / (2 * np.pi)) ** 2 * Pb2

    band1 = np.kron(np.kron(band(dim1) + band(dim1).T, eye2), eye3)
    band2 = np.kron(np.kron(eye1, band(dim2) + band(dim2).T), eye3)
    band3 = np.kron(np.kron(eye1, eye2), band(dim3) + band(dim3).T)
    Hj = -0.5 * Ej * (beta * band1 + alpha * band2 + band3)
    bandf = np.kron(np.kron(band(dim1), band(dim2)), band(dim3))

    D = alpha * b * beta + alpha * b + alpha * beta + b * beta
    Cb = alpha * b + alpha + b
    C1 = b * beta + b + beta
    C2 = alpha * b + alpha * beta + b * beta
    Ht = (4 * Ec / D) * (
        Cb * Nbt
        + C1 * Nal
        + C2 * Nb
        - 2 * b * NbtNal
        - 2 * b * beta * NalNb
        - 2 * alpha * beta * NbtNb
    )
    Ev = []
    Ek = []
    for phi in tqdm(phi_list):  # Flux dependent part
        Hf = (
            -0.5
            * b
            * Ej
            * (bandf * np.exp(2j * np.pi * phi) + bandf.T * np.exp(-2j * np.pi * phi))
        )
        Hq0 = Ht + HL + Hj + Hf
        evals0, ekets0 = np.linalg.eigh(Hq0)
        Ev.append(evals0)
        Ek.append(ekets0)
    return Ev, Ek


def circuit_spectrum_QR(Ev, Ek, params, nlist, qspace):
    Ej = params[0]
    Ec = params[1]
    Lr = params[2]
    Cr = params[3]
    alpha = params[4]
    beta = params[5]
    b = 1.0

    wr = 1e-9 / (2 * np.pi * np.sqrt(Cr * Lr))
    Ir = np.sqrt(h / (4 * np.pi * Lr * np.sqrt(Lr * Cr)))

    bn = 10
    E1 = [[] for i in range(bn)]
    nmax1 = nlist[0]
    nmax2 = nlist[1]
    nmax3 = nlist[2]
    fock = 15
    dim1 = 2 * nmax1 + 1
    dim2 = 2 * nmax2 + 1
    dim3 = 2 * nmax3 + 1

    # Matrix preparation
    eye1 = np.eye(dim1)
    eye2 = np.eye(dim2)
    eye3 = np.eye(dim3)
    eyef = np.eye(fock)

    # Resonator
    Hr0 = wr * np.matmul(dest(fock).T, dest(fock))
    Pb = np.kron(phi1_mat(nmax1), np.kron(eye2, eye3))
    Hcr = dest(fock).T + dest(fock)
    coef = -Ir * (phi0 / (2 * np.pi)) / (h * 1e9)

    for i in tqdm(range(len(Ev))):
        evals0 = Ev[i]
        ekets0 = Ek[i]
        psiq = []
        for k in range(qspace):
            psiq.append(ekets0[:, k])
        phib_mat = np.array(np.zeros((qspace, qspace), dtype=np.cdouble))
        hq0 = np.array(np.zeros((qspace, qspace), dtype=np.cdouble))
        for x in range(qspace):
            hq0[x][x] = evals0[x]
            for y in range(qspace):
                phib_mat[x][y] = np.dot(np.conjugate(psiq[x]), np.dot(Pb, psiq[y]))

        Hqco = np.kron(hq0, eyef)
        Hcco = coef * np.kron(phib_mat, Hcr)
        Hrco = np.kron(np.eye(qspace), Hr0)
        evals1, ekets1 = np.linalg.eigh(Hqco + Hcco + Hrco)

        for q in range(bn):
            E1[q].append(evals1[q])
    return E1
