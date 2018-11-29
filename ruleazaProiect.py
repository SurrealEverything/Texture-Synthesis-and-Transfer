#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:18:26 2018

@author: gabriel
"""
import cv2
from realizeazaSintezaTexturii import realizeazaSintezaTexturii
#from realizeazaTransferulTexturii import realizeazaTransferulTexturii


class parametri:
    # numele, tipul si calea texturii
    numeTextura = 'rice'
    tipTextura = 'jpg'
    caleTextura = '/home/gabriel/Spyder Projects/VA/Tema3/data/'
    # citeste textura
    textura = cv2.imread(caleTextura + numeTextura + '.' + tipTextura)

    # numele, tipul si calea imaginii
    numeImg = 'eminescu'
    tipImg = 'jpg'
    caleImg = caleTextura
    # citeste imaginea
    imagine = cv2.imread(caleImg + numeImg + '.' + tipImg)

    # de cate ori o sa fie marita imaginea
    multiplier = 2
    dimensiuneTexturaSintetizata = (textura.shape[0] * multiplier,
                                    textura.shape[1] * multiplier)

    dimensiuneBloc = 36
    nrBlocuri = 2000
    eroareTolerata = 0.1
    portiuneSuprapunere = 1/6

    # optiuni posibile: 'blocuriAleatoare', 'eroareSuprapunere',
    # 'frontieraMinima'
    metodaSinteza = 'frontieraMinima'

    # nr de iteratii pentru transferul texturii
    nrIteratii = 3

    # limita recursivitate
    # valori mari(ex: 20100) => poate rula pe blocuri mari
    # -1 => limita automata
    recLimit = 20100


def genereazaNumeTexturaSintetizata():

    if parametri.metodaSinteza == 'blocuriAleatoare':
        sinteza = 'aleator'
    elif parametri.metodaSinteza == 'eroareSuprapunere':
        sinteza = 'errSup'
    else:
        sinteza = 'frontMin'

    parametriNumerici = (
            str(parametri.multiplier) + '_'
            + str(parametri.dimensiuneBloc) + '_'
            + str(parametri.nrBlocuri) + '_'
            + str(parametri.eroareTolerata) + '_'
            + str(format(parametri.portiuneSuprapunere, '.2f')))

    nume = (parametri.numeTextura + '_' + sinteza
            + '_' + parametriNumerici + '.png')
    return nume


def genereazaNumeTexturaTransferata():

    if parametri.metodaSinteza == 'blocuriAleatoare':
        sinteza = 'aleator'
    elif parametri.metodaSinteza == 'eroareSuprapunere':
        sinteza = 'errSup'
    else:
        sinteza = 'frontMin'

    parametriNumerici = (
            str(parametri.multiplier) + '_'
            + str(parametri.dimensiuneBloc) + '_'
            + str(parametri.nrBlocuri) + '_'
            + str(parametri.eroareTolerata) + '_'
            + str(format(parametri.portiuneSuprapunere, '.2f'))
            + str(parametri.nrIteratii))

    nume = (parametri.numeImg + '_' + parametri.numeTextura + '_' + sinteza
            + '_' + parametriNumerici + '.png')
    return nume


texturaSintetizata = realizeazaSintezaTexturii(parametri)
#texturaTransferata = realizeazaTransferulTexturii(parametri)

# scriem imaginiile
cv2.imwrite(genereazaNumeTexturaSintetizata(), texturaSintetizata)
#cv2.imwrite(genereazaNumeTexturaTransferata(), texturaTransferata)
