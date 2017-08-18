"""
Author: Daniel Rueda Macias, 559207@unizar.es
Subject: Artificial Intelligence (AI) Zaragoza's University
12th September 2017
"""

# Javier Civera, jcivera@unizar.es
# December 2015
# Example of loading enron spam dataset
#--------------------------------------------------
# Juan D. Tardos, tardos@unizar.es
# 20 Dec 2016
# Tested for Python 3.5


######################################################
# Imports
######################################################

import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import KFold
import numpy as np
import json
import glob
from sklearn import metrics
import random

######################################################
# Aux. functions
######################################################

# load_enron_folder: load training, validation and test sets from an enron path
def load_enron_folder(path):

   ### Load ham mails ###

   ham_folder = path + '/ham/*.txt'
   print("Loading files:", ham_folder)
   ham_list = glob.glob(ham_folder)
   num_ham_mails = len(ham_list)
   ham_mail = []
   for i in range(0,num_ham_mails):
      ham_i_path = ham_list[i]
      #print(ham_i_path)
      ham_i_file = open(ham_i_path, 'rb')
      ham_i_str = ham_i_file.read()
      ham_i_text = ham_i_str.decode('utf-8', errors='ignore')     # Convert to Unicode
      ham_mail.append(ham_i_text)    # Append to the mail structure
      ham_i_file.close()
   random.shuffle(ham_mail)  # Random order

   # Separate into training, validation and test
   num_ham_training = int(round(0.8*num_ham_mails))
   ham_training_mail = ham_mail[0:num_ham_training]
   ham_training_labels = [0]*num_ham_training

   num_ham_validation = int(round(0.1*num_ham_mails))
   ham_validation_mail = ham_mail[num_ham_training:num_ham_training+num_ham_validation]
   ham_validation_labels = [0]*num_ham_validation

   num_ham_test = num_ham_mails - num_ham_training - num_ham_validation
   ham_test_mail = ham_mail[num_ham_training+num_ham_validation:num_ham_mails]
   ham_test_labels = [0]*num_ham_test

   print("ham mails       :", num_ham_mails)
   print("..for training  :", num_ham_training)
   print("..for validation:", num_ham_validation)
   print("..for testing   :", num_ham_test)

   ### Load spam mails ###

   spam_folder = path + '/spam/*.txt'
   print("Loading files:", spam_folder)
   spam_list = glob.glob(spam_folder)
   num_spam_mails = len(spam_list)
   spam_mail = []
   for i in range(0,num_spam_mails):
      spam_i_path = spam_list[i]
      #print(spam_i_path)
      spam_i_file = open(spam_i_path, 'rb')
      spam_i_str = spam_i_file.read()
      spam_i_text = spam_i_str.decode('utf-8', errors='ignore')     # Convert to Unicode
      spam_mail.append(spam_i_text)    # Append to the mail structure
      spam_i_file.close()
   random.shuffle(spam_mail)  # Random order

   # Separate into training, validation and test
   num_spam_training = int(round(0.8*num_spam_mails))
   spam_training_mail = spam_mail[0:num_spam_training]
   spam_training_labels = [1]*num_spam_training

   num_spam_validation = int(round(0.1*num_spam_mails))
   spam_validation_mail = spam_mail[num_spam_training:num_spam_training+num_spam_validation]
   spam_validation_labels = [1]*num_spam_validation

   num_spam_test = num_spam_mails - num_spam_training - num_spam_validation
   spam_test_mail = spam_mail[num_spam_training+num_spam_validation:num_spam_mails]
   spam_test_labels = [1]*num_spam_test

   print("spam mails      :", num_spam_mails)
   print("..for training  :", num_spam_training)
   print("..for validation:", num_spam_validation)
   print("..for testing   :", num_spam_test)

   ### spam + ham together ###
   training_mails = ham_training_mail + spam_training_mail
   training_labels = ham_training_labels + spam_training_labels
   validation_mails = ham_validation_mail + spam_validation_mail
   validation_labels = ham_validation_labels + spam_validation_labels
   test_mails = ham_test_mail + spam_test_mail
   test_labels = ham_test_labels + spam_test_labels

   data = {'training_mails': training_mails, 'training_labels': training_labels,
           'validation_mails': validation_mails, 'validation_labels':
               validation_labels, 'test_mails': test_mails,
           'test_labels': test_labels}

   return data

def kfold_cross_validation(learner, k, n, data, labels):
    """Realiza el entrenamiento del clasificador mediante el algoritmo de
        validación cruzada.
    :param learner: La distribucion que se quiere usar: Multinomial o Bernoulli
    :param k: Numero de folds para el entrenamiento del clasificador
    :param n: Numero de hiperparametros de la distribucion
    :param data: Matriz de terminos de los mails de entrenamiento
            segun el modelo de bolsa de palabras o bigramas.
    :param labels: Etiquetas de los mails de entrenamiento
    :return: Valor del suavizado de Laplace que consigue un clasificador
        mas preciso.
    """

    best_size = 0
    best_validation_error = 9999
    training_error = 0
    validation_error = 0

    if k < 2:
        print("ERROR \nk tiene que ser >= 2")
        exit(1)
    else:
        for size in range(1, n + 1):  # Para los distintos valores de los hiperparámetros
            for train_index, test_index in KFold(k).split(data):

                # Se organizan los datos segun los indices

                data_training = data[train_index[0]:train_index[-1]]
                labels_training = labels[train_index[0]:train_index[-1]]

                # Para test
                data_test = data[test_index[0]:test_index[-1]]
                labels_test = labels[test_index[0]:test_index[-1]]

                # Se prepara la distribucion que se haya especificado

                if learner == "Multinomial":
                    dist = MultinomialNB(size)
                elif learner == "Bernoulli":
                    dist = BernoulliNB(size)
                else:
                    print("ERROR \nLa distribución no coincide con "
                                    "ninguna de las esperadas")
                    exit(1)

                # Se entrena el clasificador con los datos y las etiquetas
                dist.fit(data_training, labels_training)

                # Se calcula y suma el error de los datos de entrenamiento
                # y validacion
                training_error += (1 - dist.score(data_training, labels_training))
                validation_error += (1 - dist.score(data_test, labels_test))

            training_error = training_error / k
            validation_error = validation_error / k
            if validation_error < best_validation_error:
                best_size = size
                best_validation_error = validation_error
        print("Best size: " + str(best_size))
        print("Best validation_error: " + str(best_validation_error))
    return best_size

def binary_traduction(label):
    """Traduce la etiqueta binaria de clasificación a la palabra que le corresponde.
    0 para Spam y 1 para Ham.
    :param label: Etiqueta binaria
    :return: Palabra que corresponde a la etiqueta binaria
    """
    if label == 0:
        return "Spam"
    else:
        return "Ham"

def evaluation(alpha, data, labels, data_test, labels_test, type, normalized, transformer=None):
    """Crea un clasificador con el valor del hiperparametro y evalua las
        prestaciones utilizando varias metricas. En concreto la curva de
        precision-recall, la matriz f1-score y la matriz de confusion. Para
        evaluar la distribucion normalizada es necesario llamar a la funcion
        con el parametro normalized establecido a True.
    :param alpha: Valor del hiperparametro de la distribucion
    :param data: Tupla de dos elementos:
                data[0]: Modelo de bolsa de palabras o bigramas de los mails de
                entrenamiento previamente con el lenguaje aprendido (fit())
                data[1]: Mails de entrenamiento
    :param labels: Etiquetas de los mails de entrenamiento
    :param data_test: Mails de test
    :param labels_test: Etiquetas de los mails de test
    :param type: Tipo de clasificador
    :param normalized: Indica si la bolsa de palabras esta normalizada (True) o
            no (False)
    :param transformer: Objeto para convertir la matriz de terminos a una
            distribucion normalizada. None por defecto
    :return:
    """

    if type == "Multinomial":

        # Se crea el clasificador con el alpha calculado y se entrena con la
        # matriz de terminos del documento
        classifier = MultinomialNB(alpha)
    elif type == "Bernoulli":

        # Se crea el clasificador con el alpha calculado y se entrena con la
        # matriz de terminos del documento
        classifier = BernoulliNB(alpha)
    else:
        print("ERROR \nLa distribucion no se corresponde con ninguna"
              "de las aceptadas")
        exit(1)

    if normalized:
        # Se entrena el clasificador con la matriz de terminos del vocabulario
        classifier.fit(transformer.transform(data[0].transform(data[1])), labels)

        # Se crea la matriz de terminos del documento basandose en el
        # vocabulario de los datos de entrenamiento
        test_matrix = transformer.transform(data[0].transform(data_test))
    else:
        # Se entrena el clasificador con la matriz de terminos del vocabulario
        classifier.fit(data[0].transform(data[1]), labels)

        # Se crea la matriz de terminos del documento basandose en el
        # vocabulario de los datos de entrenamiento
        test_matrix = data[0].transform(data_test)

    # Se predicen los resultados con los datos de test
    predictions = classifier.predict(test_matrix)

    # Se encuentra el primer mail que haya clasificado mal
    found = False
    i = 0
    while (not found) and (i <= len(labels_test)):
        if predictions[i] != labels_test[i]:
            print()
            print("Se ha predecido como " + binary_traduction(predictions[i])
                  + " y realmente es " + binary_traduction(labels_test[i]) + ".")
            print()
            print("------ Mail ------")
            print(data_test[i])
            found = True
        else:
            i += 1

    # Se predicen la probabilidad de pertenecer a cada clase con
    # los datos de test
    probabilities = classifier.predict_proba(test_matrix)
    # print()
    # print("--- Probabilidades ---")
    # print(probabilities)

    # Se imprimen por pantalla la curva precision-recall,
    # f1-score y la matriz de confusion
    # Curva precision recall
    precisionRecall_curve(labels_test, predictions)

    # Matriz de confusion
    confusion_matrix(labels_test, predictions)

    # f1-score
    print()
    print("F1-score: " + str(metrics.f1_score(labels_test, predictions)))

def precisionRecall_curve(test_labels, predictions):
    """Muestra por pantalla la curva Precision-Recall.
    :param test_labels: Etiquetas a testear
    :param predictions: Etiquetas predecidas por el clasificador
    :return:
    """

    precision, recall, thresholds = metrics.precision_recall_curve(test_labels,
                                                                   predictions)
    plt.clf()
    plt.title("Precision-Recall curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.plot(precision, recall)
    plt.show()

def confusion_matrix(test_labels, predictions):
    """Muestra por pantalla la matriz de confusion.
    :param test_labels: etiquetas a testear
    :param predictions: etiquetas predecidas por el clasificador
    :return:
    """

    confusion_matrix = metrics.confusion_matrix(test_labels, predictions)
    print()
    print("Confusion matrix")
    print("      spam     ham")
    print("spam  %d      %d" % (confusion_matrix[0][0], confusion_matrix[0][1]))
    print(" ham  %d      %d" % (confusion_matrix[1][0], confusion_matrix[1][1]))

def prueba(bag_of, normalized, classifier, folds, data_mails, data_lables, test_mails, test_labels):
    """Realiza un entrenamiento y evaluacion de un clasificador eligiendo una
    distribucion de probabilidad determinada dado un modelo de bolsa de palabras,
    pudiendo este ser normalizado o no
    :param bag_of: Bolsa de palabras (words) o bigramas (bigrams)
    :param normalized: Normalizar el modelo (True) o no (False)
    :param classifier: Tipo de distribucion para construir el clasificador
    (Multinomial o Bernoulli)
    :param folds: Numero de folds para ejecutar el algoritmo kfold_cross_validation
    :param data_mails: Mails de datos
    :param data_lables: Etiquetas de los mails de datos
    :param test_mails: Mails de test
    :param test_labels: Etiquetas de los mails de test
    :return:
    """
    bag_type = "undefined"
    distribution = "undefined"
    model_bag = "indefinido"
    if bag_of == "words":
        model_bag = "unigramas"
        bag_type = CountVectorizer(ngram_range=(1, 1))  # Bolsa de palabras
    elif bag_of == "bigrams":
        model_bag = "bigramas"
        bag_type = CountVectorizer(ngram_range=(1, 2))  # Bolsa de bigramas
    else:
        print("ERROR \nVuelva a introducir el modelo de bolsa de palabras que"
              "desea")
        exit(1)

    if classifier == "Multinomial":
        distribution = "Multinomial"
    elif classifier == "Bernoulli":
        distribution = "Bernoulli"
    else:
        print("ERROR \nVuelva a introducir la dsitribucion de probabilidad con"
              "la que quiere crear el clasificador")
        exit(1)

    if normalized:
        print()
        print("---------------------------------------------------------------")
        print("                %s %s con %s               " % (distribution, "normalizada", model_bag))
        print("---------------------------------------------------------------")
        transformer = TfidfTransformer()
        alpha = kfold_cross_validation(distribution, folds, 1,
                                       transformer.fit_transform(bag_type.fit_transform(data_mails)),
                                       data_lables)
        evaluation(alpha, (bag_type.fit(data_mails), data_mails), data_lables,
                   test_mails, test_labels, distribution, True, transformer)
    else:
        print()
        print("---------------------------------------------------------------")
        print("                %s %s con %s               " % (distribution, "sin normalizar", model_bag))
        print("---------------------------------------------------------------")
        alpha = kfold_cross_validation(distribution, folds, 1,
                                       bag_type.fit_transform(data_mails),
                                       data_lables)
        evaluation(alpha, (bag_type.fit(data_mails), data_mails), data_lables,
                   test_mails, test_labels, distribution, False)


######################################################
# Main
######################################################

# Ruta a la carpeta donde se encuentran las carpetas enron que contienen los mails
# para hacer las pruebas.
# Nota: De Linux a Windows las / se cambian por \

if len(sys.argv) < 3:
    print("ERROR \nIntroduzca la ruta a los mails Enron y el número de folds")
    exit(1)
else:
    path = sys.argv[1]
    print("Starting...")

    # Path to the folder containing the mails
    folder_enron1 = path + 'enron1'
    folder_enron2 = path + 'enron2'
    folder_enron3 = path + 'enron3'
    folder_enron4 = path + 'enron4'
    folder_enron5 = path + 'enron5'
    folder_enron6 = path + 'enron6'

    # Load mails
    data1 = load_enron_folder(folder_enron1)
    data2 = load_enron_folder(folder_enron2)
    data3 = load_enron_folder(folder_enron3)
    data4 = load_enron_folder(folder_enron4)
    data5 = load_enron_folder(folder_enron5)


    # Prepare data
    training_mails = data1['training_mails']+data2['training_mails'] + \
                     data3['training_mails']+data4['training_mails'] + \
                     data5['training_mails']
    training_labels = data1['training_labels']+data2['training_labels'] + \
                      data3['training_labels']+data4['training_labels'] + \
                      data5['training_labels']
    validation_mails = data1['validation_mails']+data2['validation_mails'] + \
                       data3['validation_mails']+data4['validation_mails'] + \
                       data5['validation_mails']
    validation_labels = data1['validation_labels']+data2['validation_labels'] + \
                        data3['validation_labels']+data4['validation_labels'] + \
                        data5['validation_labels']

    # Loading test data
    data6 = load_enron_folder(folder_enron6)
    test_mails = data6['test_mails']
    test_labels = data6['test_labels']


    ################################################################################
    # BATERIA DE PRUEBAS
    ################################################################################

    folds = int(sys.argv[2])  # Para ir probando a ver cuantos folds son mejores

    # Multinomial unigramas
    prueba("words", False, "Multinomial", folds, training_mails + validation_mails,
           training_labels + validation_labels, test_mails, test_labels)

    # Multinomial bigramas
    #prueba("bigrams", False, "Multinomial", folds, training_mails + validation_mails,
    #       training_labels + validation_labels, test_mails, test_labels)

    # Multinomial normalizada unigramas
    #prueba("words", True, "Multinomial", folds, training_mails + validation_mails,
    #       training_labels + validation_labels, test_mails, test_labels)

    # Multinomial normalizada bigramas
    #prueba("bigrams", True, "Multinomial", folds, training_mails + validation_mails,
    #       training_labels + validation_labels, test_mails, test_labels)

    # Bernoulli unigramas
    #prueba("words", False, "Bernoulli", folds, training_mails + validation_mails,
    #       training_labels + validation_labels, test_mails, test_labels)

    # Bernoulli bigramas
    #prueba("bigrams", False, "Bernoulli", folds, training_mails + validation_mails,
    #       training_labels + validation_labels, test_mails, test_labels)

    # Bernoulli normalizada unigramas
    #prueba("words", True, "Bernoulli", folds, training_mails + validation_mails,
    #       training_labels + validation_labels, test_mails, test_labels)

    # Bernoulli normalizada bigramas
    #prueba("bigrams", True, "Bernoulli", folds, training_mails + validation_mails,
    #       training_labels + validation_labels, test_mails, test_labels)


