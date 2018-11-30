#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:29:21 2018

@author: gabriel
"""
import numpy as np
import cv2
import math
from itertools import product
from functiiUtile import (genereazaBlocuri, calculeazaDistantaSuprapunere,
                          gasesteDrumMinim, floodFillIterativ,
                          calculeazaDistantaPatch)


def realizeazaTransferulTexturii(parametri):

    nrIteratii = parametri.nrIteratii

    HT, WT, c = parametri.textura.shape

    nrBlocuri = parametri.nrBlocuri

    HI, WI, c = parametri.img.shape

    eroareTolerata = parametri.eroareTolerata + 1

    inf = np.iinfo(np.uint16).max
    dimYmin = inf
    dimXmin = inf

    # factorul cu care vom mari un bloc(intr-o limita de resurse), pentru a
    # compensa micsorarea ulterioara
    factorMarire = (3 ** min(nrIteratii-1, parametri.limitaMarire))
    dimBloc = parametri.dimensiuneBloc * factorMarire
    # daca textura initiala nu e suficient de mare
    if HT / dimBloc < 8:
        # marim textura initiala, pentru a permite functionarea unui
        # numar mai mare de iteratii
        HTR = HT * factorMarire
        WTR = WT * factorMarire
        textura = cv2.resize(parametri.textura, (HTR, WTR), cv2.INTER_CUBIC)
    else:
        factorMarire = 1

    for iteratie in range(nrIteratii):

        dimSuprapunere = math.floor(parametri.portiuneSuprapunere * dimBloc)

        blocuri = genereazaBlocuri(textura, nrBlocuri,  dimBloc)

        # aflam numarul de blocuri pentru imaginea mare
        nrBlocuriY = math.ceil(HI / dimBloc)
        nrBlocuriX = math.ceil(WI / dimBloc)

        # calculam dimensiunile imaginii tinand cont de suprapunere
        dimY = nrBlocuriY * (dimBloc - dimSuprapunere) + dimSuprapunere
        dimX = nrBlocuriX * (dimBloc - dimSuprapunere) + dimSuprapunere

        if dimY < dimYmin:
            dimYmin = dimY
        if dimX < dimXmin:
            dimXmin = dimX

        if iteratie == 0:
            texturaTransferata = np.zeros((dimY, dimX, 3), np.uint8)

        # actualizam ponderea dupa aceasta formula
        if nrIteratii > 1:
            pondere = 0.8 * iteratie/(nrIteratii-1) + 0.1
        else:
            pondere = 0

        for y, x in product(range(nrBlocuriY), range(nrBlocuriX)):

            # calculam indicii plasarii blocului suprapus
            startY = y * (dimBloc - dimSuprapunere)
            startX = x * (dimBloc - dimSuprapunere)
            endY = startY + dimBloc
            endX = startX + dimBloc

            # distanta totala de suprapunere pentru fiecare bloc
            distante = np.zeros((nrBlocuri,))

            if iteratie > 1:
                if y:
                    # calculam suprapunerea orizontala
                    suprapunereImagineY = texturaTransferata[
                        startY : startY + dimSuprapunere,
                        startX : endX,
                        :]

                    # aflam distanta pentru portiunea suprapusa orizontal dintre
                    # imaginea sintetizata pana acum si toate celelalte blocuri
                    distanteY = calculeazaDistantaSuprapunere(
                                blocuri, suprapunereImagineY, dimSuprapunere, 'Y')

                if x:
                    # calculam suprapunerea verticala
                    suprapunereImagineX = texturaTransferata[
                        startY : endY,
                        startX : startX + dimSuprapunere,
                        :]

                    # aflam distanta pentru portiunea suprapusa vertical dintre
                    # imaginea sintetizata pana acum si toate celelalte blocuri
                    distanteX = calculeazaDistantaSuprapunere(
                                blocuri, suprapunereImagineX, dimSuprapunere, 'X')

                if y and x:
                    # calculam suprapunerea comuna
                    suprapunereImagineXY = texturaTransferata[
                        startY : startY + dimSuprapunere,
                        startX : startX + dimSuprapunere,
                        :]

                    # aflam distanta pentru portiunea suprapusa diagonal dintre
                    # imaginea sintetizata pana acum si toate celelalte blocuri
                    distanteXY = calculeazaDistantaSuprapunere(
                                blocuri, suprapunereImagineXY, dimSuprapunere, 'XY')

                if y:
                    # adaugam suprapunerea orizontala
                    distante += distanteY
                if x:
                    # adaugam suprapunerea verticala
                    distante += distanteX
                if y and x:
                    # scadem suprapunerea comuna, ca sa nu fie calculata de 2 ori
                    distante -= distanteXY

            patchSursa = parametri.img[
                        startY : endY,
                        startX : endX,
                        :]

            intensitate = calculeazaDistantaPatch(blocuri, patchSursa)

            distante = pondere * distante + (1-pondere) * intensitate

            # aflam distanta minima dintre toate distantele
            bestMatch = distante.min()

            # aflam o multime de indici cu o distanta aproapiata de cea minima
            # cu scopul de a varia blocurile folosite
            indici = np.flatnonzero(distante <= eroareTolerata * bestMatch)

            # alegem aleator un indice din acea multime
            indice = np.random.choice(indici)

            mascaSuprapunere = np.zeros((dimBloc, dimBloc), np.uint8)

            if y:
                # partea orizontala de suprapus a imaginii
                oldY = texturaTransferata[
                            startY : startY + dimSuprapunere,
                            startX : endX,
                            :]
                # partea verticala  a blocului care urmeaza sa suprapuna
                newY = blocuri[:dimSuprapunere, :, :, indice]
                # matricea de cost al fiecarei suprapuneri
                EY = np.power((oldY - newY), 2)
                EY = EY[:, :, 0] + EY[:, :, 1] + EY[:, :, 2]
                # frontiera de cost minim
                drumMinimY = gasesteDrumMinim(EY)

                for i in range(dimBloc):
                    mascaSuprapunere[drumMinimY[i], i] += 1

            if x:
                # partea orizontala de suprapus a imaginii
                oldX = texturaTransferata[
                            startY : endY,
                            startX : startX + dimSuprapunere,
                            :]
                # partea verticala  a blocului care urmeaza sa suprapuna
                newX = blocuri[:, :dimSuprapunere, :, indice]
                # matricea de cost al fiecarei suprapuneri
                EX = np.power((oldX - newX), 2)
                EX = EX[:, :, 0] + EX[:, :, 1] + EX[:, :, 2]
                # frontiera de cost minim
                drumMinimX = gasesteDrumMinim(EX)

                for i in range(dimBloc):
                    mascaSuprapunere[i, drumMinimX[i]] += 1

            if x and y:

                # adaugam 2 si in zona in care drumurile se
                # intersecteaza pe diagonala
                for i, j in product(
                        range(dimSuprapunere-1), range(dimSuprapunere-1)):
                    if (mascaSuprapunere[i][j]
                            and mascaSuprapunere[i+1][j]
                            and mascaSuprapunere[i][j+1]
                            and mascaSuprapunere[i+1][j+1]):

                            mascaSuprapunere[i][j] = 2
                            mascaSuprapunere[i+1][j] = 2
                            mascaSuprapunere[i][j+1] = 2
                            mascaSuprapunere[i+1][j+1] = 2

                # stergem drumul y pana la intersectia cu x
                for i in range(dimBloc):
                    j = drumMinimY[i]
                    if mascaSuprapunere[j, i] == 2:
                        indY = i
                        break
                    else:
                        mascaSuprapunere[j, i] = 0

                # stergem drumul x pana la intersectia cu y
                for i in range(dimBloc):
                    j = drumMinimX[i]
                    if mascaSuprapunere[i, j] == 2:
                        indX = i
                        break
                    else:
                        mascaSuprapunere[i, j] = 0

                # inlocuim 2 urile cu 1
                for i in range(indY, dimBloc):
                    j = drumMinimY[i]
                    if mascaSuprapunere[j, i] == 2:
                        mascaSuprapunere[j, i] = 1

                for i in range(indX, dimBloc):
                    j = drumMinimX[i]
                    if mascaSuprapunere[i, j] == 2:
                        mascaSuprapunere[i, j] = 1

            # Umple cu 1 partea care urmeaza sa fie scrisa
            floodFillIterativ(mascaSuprapunere, dimBloc-1, dimBloc-1)

            # partea de suprapus a imaginii sintetizate pana acum
            old = texturaTransferata[startY:endY, startX:endX, :]

            # blocul care urmeaza sa suprapuna imaginea
            new = blocuri[:, :, :, indice]

            mascaNegata = np.logical_not(mascaSuprapunere)

            # transformam mastile la dimensiunea unei imagini
            mascaSuprapunere = np.repeat(
                    mascaSuprapunere[:, :, np.newaxis], 3, axis=2)
            mascaNegata = np.repeat(
                    mascaNegata[:, :, np.newaxis], 3, axis=2)

            # partea imaginii sintetizate care nu va fi suprapusa
            old = old * mascaNegata
            # partea blocului care suprapune imaginea
            new = new * mascaSuprapunere
            rezultat = old + new

            texturaTransferata[
                        startY : endY,
                        startX : endX,
                        :] = rezultat

        if iteratie != nrIteratii-1:
            # actualizam dimensiunea blocului pentru urmatoarea iteratie
            dimBloc = math.floor(dimBloc/3)

            if factorMarire > 1:
                # micsoram textura originala, pentru a avea blocuri proportionale
                HTR = math.floor(HTR/3)
                WTR = math.floor(WTR/3)
                textura = cv2.resize(textura, (HTR, WTR), cv2.INTER_CUBIC)
            factorMarire = factorMarire/3

    return texturaTransferata[:dimYmin, :dimXmin, :]
