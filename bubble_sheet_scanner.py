from PIL import Image
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import pandas as pd
import xlrd
import os

#Define o Gabarito

ANSWER_KEY = pd.read_excel("C:/Users/rafae/Documents/Leitor/Gabarito.xlsx")
ANSWER_KEY = dict(zip(ANSWER_KEY['Questao'], ANSWER_KEY['Resposta']))

for i in range (1,len(ANSWER_KEY)+1):
    if ANSWER_KEY[i] == 'A':
        ANSWER_KEY[i] = 0
    if ANSWER_KEY[i] == 'B':
        ANSWER_KEY[i] = 1
    if ANSWER_KEY[i] == 'C':
        ANSWER_KEY[i] = 2
    if ANSWER_KEY[i] == 'D':
        ANSWER_KEY[i] = 3
    if ANSWER_KEY[i] == 'E':
        ANSWER_KEY[i] = 4

dictAnswer = []
dictAnswer.append({k: ANSWER_KEY[k] for k in range(1,26)})
dictAnswer.append({k: ANSWER_KEY[k] for k in range(26,51)})
dictAnswer.append({k: ANSWER_KEY[k] for k in range(51,71)})

Resultado = pd.DataFrame(columns=['RA','Nome','Fisica', 'Portugues', 'Ingles', 'Matematica', 'Quimica', 'Total'])

for nameFile in os.listdir("C:/Users/rafae/Documents/Leitor/Cartoes"):

    img = Image.open(r"C:/Users/rafae/Documents/Leitor/Cartoes/" + nameFile)
    
    ##Coluna 1

    width, height = img.size 

    left = (1.5)*width/13
    top = (13.5)*height/33
    right = (2.7)*width/11 + 0.2
    bottom = (22.7)*height/26 + 0.3
    
    col1 = img.crop((left, top, right, bottom))
    
    col1.save("C:/Users/rafae/Documents/Leitor/parciais/col1.jpg")

    ##Coluna 2

    left = (4.3)*width/13
    top = (13.5)*height/33
    right = (0.95)*width/2 + 0.2
    bottom = (22.7)*height/26 + 0.3
    
    col2 = img.crop((left, top, right, bottom))
    
    col2.save("C:/Users/rafae/Documents/Leitor/parciais/col2.jpg")

    ##Coluna 3
    
    left = (1.1)*width/2
    top = (13.5)*height/33
    right = (8.9)*width/13 + 0.1
    bottom = (20.3)*height/26 + 0.3
    
    col3 = img.crop((left, top, right, bottom))
    
    col3.save("C:/Users/rafae/Documents/Leitor/parciais/col3.jpg")

    ##Matricula
    
    left = (1.5)*width/11
    top = height/5 + 0.1
    right = (2.09)*width/5 + 0.3
    bottom = (9.3)*height/26 + 0.3
    
    mat = img.crop((left, top, right, bottom))
    
    mat.save("C:/Users/rafae/Documents/Leitor/parciais/mat.jpg")

    #Ler a matricula

    img = cv2.imread("C:/Users/rafae/Documents/Leitor/parciais/mat.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

    matricula = ""

    for (q, i) in enumerate(np.arange(0, len(questionCnts), 10)):
        color = (0, 0, 255)

        cnts = contours.sort_contours(questionCnts[i:i + 10])[0]
        bubbled = None

        for (j, c) in enumerate(cnts):
            
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # cv2.imshow("image", mask)
            # cv2.waitKey(0)

            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
                
        matricula = matricula + str(bubbled[1])
        color = (0, 255, 0)

    correct = [0,0,0]
    F = Q = M = I = P = 0
    e = [0, 25, 50]
    numberOfQuestions = 0

    for t in range(0,3):

        img = cv2.imread("C:/Users/rafae/Documents/Leitor/parciais/col"+str(t+1)+".jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
                questionCnts.append(c)

        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

        for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
            cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
            bubbled = None

            for (j, c) in enumerate(cnts):
                
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)

                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)
                    
            color = (0, 0, 255)
            k = dictAnswer[t][q + e[t] + 1]
            
            if k == bubbled[1]:
                color = (0, 255, 0)
                correct[t] += 1
            
            #REGRAS EXTRAS
            if k == 'A/C' and (bubbled[1]==0 or bubbled[1]==2):
                color = (0, 255, 0)
                correct[t] += 1
            
            if k == 'X':
                correct[t] += 1

            #POR MATERIA
            if q + e[t] + 1 == 15:
                F = correct[t]
            if q + e[t] + 1 == 25:
                P = correct[t] - F
            if q + e[t] + 1 == 30:
                aux = correct[t]
                P += correct[t]
            if q + e[t] + 1 == 40:
                I = correct[t] - aux
            if q + e[t] + 1 == 50:
                M = correct[t] - I - aux
            if q + e[t] + 1 == 55:
                aux = correct[t]
                M += correct[t]
            if q + e[t] + 1 == 70:
                Q = correct[t] - aux
    T = F + P + M + Q
    #print(F, P, I, M, Q, T)
    Resultado = Resultado.append({'RA': matricula,'Nome': "",'Fisica': F, 'Portugues': P, 'Ingles': I, 'Matematica': M, 'Quimica': Q, 'Total': T}, ignore_index=True)
Resultado.to_csv(
        "C:/Users/rafae/Documents/Leitor/Resultado.csv",
        sep = ";",
        index = False
    )