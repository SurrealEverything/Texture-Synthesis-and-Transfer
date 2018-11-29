#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:12:14 2018

@author: gabriel
"""
import numpy as np


def genereazaBlocuri(textura, nrBlocuri,  dimBloc):

    H, W, c = textura.shape

    # o imagine este o matrice cu 3 dimensiuni: inaltime x latime x nrCanale
    # variabila blocuri - matrice cu 4 dimensiuni: punem fiecare bloc
    # (portiune din textura initiala) unul peste altul
    blocuri = np.empty((dimBloc, dimBloc, 3, nrBlocuri), np.uint8)

    # selecteaza blocuri aleatoare din textura initiala
    # genereaza (in maniera vectoriala) punctul din stanga sus al blocurilor
    y = np.random.randint(H-dimBloc, size=nrBlocuri)
    x = np.random.randint(W-dimBloc, size=nrBlocuri)

    # extrage portiunea din textura initiala continand blocul
    for i in range(nrBlocuri):
        blocuri[:, :, :, i] = textura[
                y[i] : y[i] + dimBloc,
                x[i] : x[i] + dimBloc,
                :]

    return blocuri


def calculeazaDistantaSuprapunere(blocuri, suprapunereImagine, w, caz):
    """aflam distanta euclidiana pentru portiunea suprapusa dintre
    imaginea sintetizata pana acum si toate celelalte blocuri
    """
    nrBlocuri = blocuri.shape[3]
    distante = np.empty((nrBlocuri,))

    a = suprapunereImagine

    for i in range(nrBlocuri):
        # alegem forma corecta a suprapunerii blocului
        if caz == 'X':
            b = blocuri[:, :w, :, i]
        elif caz == 'Y':
            b = blocuri[:w, :, :, i]
        elif caz == 'XY':
            b = blocuri[:w, :w, :, i]
        # calculam distanta euclidiana
        # distante[i] = np.linalg.norm(a-b)
        distante[i] = np.sqrt(np.sum(np.square(a-b)))

    return distante


def calculeazaDistantaPatch(blocuri, patch):
    """aflam distanta euclidiana dintre patch-ul curent din imagine
    si toate blocurile
    """
    nrBlocuri = blocuri.shape[3]
    distante = np.empty((nrBlocuri,))

    a = patch

    for i in range(nrBlocuri):
        b = blocuri[:, :, :, i]
        # calculam distanta euclidiana
        distante[i] = np.linalg.norm(a-b)
        # distante[i] = np.sqrt(np.sum(np.square(a-b)))

    return distante

def gasesteDrumMinim(E):
    """"gaseste frontiera de cost minim"""

    rotit = 0
    h, w = E.shape
    # daca suprapunerea este orizontala, rotim imaginea
    if h < w:
        E = np.rot90(E)
        h, w = w, h
        rotit = 1

    drumMinim = np.empty((h,), np.uint16)
    inf = np.iinfo(np.uint32).max
    costE = np.full((h, w), inf, np.uint32)
    costE[0, :] = E[0, :]

    # calculam costul pentru fiecare pixel
    for i in range(1, h):
        for j in range(w):
            st = j-1
            mij = j
            dr = j+1

            if j == 0:
                costE[i, j] = costE[i-1, [mij, dr]].min() + E[i, j]
            elif j == w-1:
                costE[i, j] = costE[i-1, [st, mij]].min() + E[i, j]
            else:
                costE[i, j] = costE[i-1, [st, mij, dr]].min() + E[i, j]

    drumMinim[h-1] = costE[h-1].argmin()

    # reconstituim traseul de cost minim
    for i in range(h-2, -1, -1):

        st = drumMinim[i+1] - 1
        mij = drumMinim[i+1]
        dr = drumMinim[i+1] + 1

        if mij == 0:
            optiune = costE[i, [mij, dr]].argmin()
        elif mij == w - 1:
            optiune = costE[i, [st, mij]].argmin() - 1
        else:
            optiune = costE[i, [st, mij, dr]].argmin() - 1

        drumMinim[i] = mij + optiune

    # daca imaginea a fost rotita, recalculam indicii drumului
    if rotit:
        dFixed = np.empty((h,), np.uint16)
        for i in range(h):
            j = drumMinim[i]
            dFixed[h-1-i] = j

        drumMinim = dFixed

    return drumMinim


def floodFill(masca, x, y):
    """Umple cu 1 partea care urmeaza sa fie scrisa"""
    dimBloc = masca.shape[0]
    if masca[x][y] == 0:
        masca[x][y] = 1
        # invocam functia recursiv:
        if x > 0:
            floodFill(masca, x-1, y)
        if x < dimBloc-1:
            floodFill(masca, x+1, y)
        if y > 0:
            floodFill(masca, x, y-1)
        if y < dimBloc-1:
            floodFill(masca, x, y+1)
