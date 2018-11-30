#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:17:56 2018

@author: gabriel
"""
import numpy as np
import math
from itertools import product
from functiiUtile import (genereazaBlocuri, calculeazaDistantaSuprapunere,
                          gasesteDrumMinim, floodFillIterativ)


def realizeazaSintezaTexturii(parametri):

    dimBloc = parametri.dimensiuneBloc
    nrBlocuri = parametri.nrBlocuri

    HT = parametri.dimensiuneTexturaSintetizata[0]
    WT = parametri.dimensiuneTexturaSintetizata[1]

    eroareTolerata = parametri.eroareTolerata + 1

    dimSuprapunere = math.floor(parametri.portiuneSuprapunere * dimBloc)

    blocuri = genereazaBlocuri(parametri.textura, nrBlocuri,  dimBloc)

    # aflam numarul de blocuri pentru imaginea mare
    nrBlocuriY = math.ceil(HT / dimBloc)
    nrBlocuriX = math.ceil(WT / dimBloc)

    if parametri.metodaSinteza == 'blocuriAleatoare':
        # completeaza imaginea de obtinut cu blocuri aleatoare

        texturaSintetizata = np.empty((HT, WT, 3), np.uint8)
        texturaSintetizataMaiMare = np.empty(
            (nrBlocuriY * dimBloc, nrBlocuriX * dimBloc, 3),
            np.uint8)

        for y, x in product(range(nrBlocuriY), range(nrBlocuriX)):

            # se alege un bloc aleator
            indice = np.random.randint(nrBlocuri)

            # adaugam acel bloc in imagine
            texturaSintetizataMaiMare[
                y * dimBloc : (y+1) * dimBloc,
                x * dimBloc : (x+1) * dimBloc,
                :] = blocuri[:, :, :, indice]

        # cropam imaginea mare la dimensiunea dorita
        texturaSintetizata = texturaSintetizataMaiMare[:HT, :WT, :]

    else:
        # completeaza imaginea de obtinut cu blocuri ales
        # in functie de eroare de suprapunere

        # calculam dimensiunile imaginii tinand cont de suprapunere
        dimY = nrBlocuriY * (dimBloc - dimSuprapunere) + dimSuprapunere
        dimX = nrBlocuriX * (dimBloc - dimSuprapunere) + dimSuprapunere
        texturaSintetizata = np.zeros((dimY, dimX, 3), np.uint8)

        for y, x in product(range(nrBlocuriY), range(nrBlocuriX)):

            # calculam indicii plasarii blocului suprapus
            startY = y * (dimBloc - dimSuprapunere)
            startX = x * (dimBloc - dimSuprapunere)
            endY = startY + dimBloc
            endX = startX + dimBloc

            if y == 0 and x == 0:
                # daca e primul bloc, il alegem aleator
                indice = np.random.randint(nrBlocuri)

                texturaSintetizata[
                    startY : endY,
                    startX : endX,
                    :] = blocuri[:, :, :, indice]

                continue

            if y:
                # calculam suprapunerea orizontala
                suprapunereImagineY = texturaSintetizata[
                    startY : startY + dimSuprapunere,
                    startX : endX,
                    :]

                # aflam distanta pentru portiunea suprapusa orizontal dintre
                # imaginea sintetizata pana acum si toate celelalte blocuri
                distanteY = calculeazaDistantaSuprapunere(
                            blocuri, suprapunereImagineY, dimSuprapunere, 'Y')

            if x:
                # calculam suprapunerea verticala
                suprapunereImagineX = texturaSintetizata[
                    startY : endY,
                    startX : startX + dimSuprapunere,
                    :]

                # aflam distanta pentru portiunea suprapusa vertical dintre
                # imaginea sintetizata pana acum si toate celelalte blocuri
                distanteX = calculeazaDistantaSuprapunere(
                            blocuri, suprapunereImagineX, dimSuprapunere, 'X')

            if y and x:
                # calculam suprapunerea comuna
                suprapunereImagineXY = texturaSintetizata[
                    startY : startY + dimSuprapunere,
                    startX : startX + dimSuprapunere,
                    :]

                # aflam distanta pentru portiunea suprapusa diagonal dintre
                # imaginea sintetizata pana acum si toate celelalte blocuri
                distanteXY = calculeazaDistantaSuprapunere(
                            blocuri, suprapunereImagineXY, dimSuprapunere, 'XY')

            # distanta totala de suprapunere pentru fiecare bloc
            distante = np.zeros((nrBlocuri,))

            if y:
                # adaugam suprapunerea orizontala
                distante += distanteY
            if x:
                # adaugam suprapunerea verticala
                distante += distanteX
            if y and x:
                # scadem suprapunerea comuna, ca sa nu fie calculata de 2 ori
                distante -= distanteXY

            # aflam distanta minima dintre toate distantele
            bestMatch = distante.min()

            # aflam o multime de indici cu o distanta aproapiata de cea minima
            # cu scopul de a varia blocurile folosite
            indici = np.flatnonzero(distante <= eroareTolerata * bestMatch)

            # alegem aleator un indice din acea multime
            indice = np.random.choice(indici)

            if parametri.metodaSinteza == 'eroareSuprapunere':
                # suprapunem blocurile in totalitate

                texturaSintetizata[
                        startY : endY,
                        startX : endX,
                        :] = blocuri[:, :, :, indice]

            elif parametri.metodaSinteza == 'frontieraMinima':
                # suprapunem blocurile la o frontiera de cost minim

                mascaSuprapunere = np.zeros((dimBloc, dimBloc), np.uint8)

                if y:
                    # partea orizontala de suprapus a imaginii
                    oldY = texturaSintetizata[
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
                    oldX = texturaSintetizata[
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
                old = texturaSintetizata[startY:endY, startX:endX, :]

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

                texturaSintetizata[
                            startY : endY,
                            startX : endX,
                            :] = rezultat
    return texturaSintetizata
