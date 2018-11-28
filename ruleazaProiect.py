#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:18:26 2018

@author: gabriel
"""
import cv2
import numpy as np
from realizeazaSintezaTexturii import realizeazaSintezaTexturii


class parametri:
    # puteti inlocui numele imaginii
    numeImg = 'radishes'
    tipImg = 'jpg'
    # citeste imaginea care va fi transformata in mozaic
    img = cv2.imread('/home/gabriel/Spyder Projects/VA/Tema3/data/'
                     + numeImg + '.' + tipImg)

    texturaInitiala = np.array(img, copy=True)

    multiplier = 2
    dimensiuneTexturaSintetizata = (texturaInitiala.shape[0] * multiplier,
                                    texturaInitiala.shape[1] * multiplier)

    dimensiuneBloc = 36*2
    nrBlocuri = 2000
    eroareTolerata = 0.1
    portiuneSuprapunere = 1/6

    # optiuni posibile: 'blocuriAleatoare', 'eroareSuprapunere',
    # 'frontieraMinima'
    metodaSinteza = 'frontieraMinima'

    # limita recursivitate (valori mari => poate rula pe blocuri mari)
    recLimit = 20100

def genereazaNume():

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

    nume = (parametri.numeImg + '_' + sinteza
            + '_' + parametriNumerici + '.png')
    return nume


imgSintetizata = realizeazaSintezaTexturii(parametri)

# scriem imaginea
cv2.imwrite(genereazaNume(), imgSintetizata)
