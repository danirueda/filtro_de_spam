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

"""
    Realiza el entrenamiento del clasificador mediante el algoritmo de
    validación cruzada.
        learner: La distribucion que se quiere usar: Multinomial o Bernoulli
        k: Numero de folds para el entrenamiento del clasificador
        n: Numero de hiperparametros de la distribucion
        data: Matriz de terminos de los mails de entrenamiento 
        segun el modelo de bolsa de palabras o bigramas.
        labels: Etiquetas de los mails de entrenamiento
"""
def kfold_cross_validation(learner, k, n, data, labels):
    best_size = 0
    best_validation_error = 9999
    training_error = 0
    validation_error = 0

    for size in range(1, n + 1): # Para los distintos valores de los hiperparámetros
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
                print("ERROR")
                raise NameError("ERROR \n La distribución no coincide con "
                                "ninguna de las esperadas")

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

"""
    Crea un clasificador con el valor del hiperparametro y evalua las
    prestaciones utilizando varias metricas. En concreto la curva de
    precision-recall, la matriz f1-score y la matriz de confusion.
        alpha: Valor del hiperparametro de la distribucion
        data: Tupla de dos elementos que contiene el modelo de bolsa de palabras
        o bigramas de los mails de entrenamiento y los mails de entrenamiento
        respectivamente
        labels: Etiquetas de los mails de entrenamiento
        data_test: Mails de test
        labels_test: Etiquetas de los datos de test
        type: Tipo de clasificador
"""
def evaluation(alpha, data, labels, data_test, labels_test, type):
    if type == "Multinomial":

        # Se crea el clasificador con el alpha calculado y se entrena con la
        # matriz de terminos del documento
        classifier = MultinomialNB(alpha)
        classifier.fit(data[0].transform(data[1]), labels)


        # Se crea la matriz de terminos del documento basandose en el
        # vocabulario de los datos de entrenamiento
        test_matrix = data[0].transform(data_test)

        # Se predicen los resultados con los datos de test
        print("datos de test: " + str(test_matrix))
        predictions = classifier.predict(test_matrix)

        # Se imprimen por pantalla la curva precision-recall,
        # f1-score y la matriz de confusion
        metrics.precision_recall_curve(labels_test, predictions)
    elif type == "Bernoulli":
        # Se crea el clasificador con el alpha calculado y se entrena
        classifier = BernoulliNB(alpha)
        classifier.fit(data, labels)

        # Se predicen los resultados con los datos de test
        predictions = classifier.predict(data_test)

        # Se imprimen por pantalla la curva precision-recall,
        # f1-score y la matriz de confusion
        metrics.precision_recall_curve(labels_test, predictions)
    else:
        print("ERROR \n La distribucion no se corresponde con ninguna"
              "de las aceptadas")

######################################################
# Main
######################################################

print("Starting...")

# Path to the folder containing the mails
folder_enron1 = '/home/dani/Escritorio/Unizar/IA/TP6s/TP6_filtro_de_spam/' \
                'mails/enron1'
folder_enron2 = '/home/dani/Escritorio/Unizar/IA/TP6s/TP6_filtro_de_spam/' \
                'mails/enron2'
folder_enron3 = '/home/dani/Escritorio/Unizar/IA/TP6s/TP6_filtro_de_spam/' \
                'mails/enron3'
folder_enron4 = '/home/dani/Escritorio/Unizar/IA/TP6s/TP6_filtro_de_spam/' \
                'mails/enron4'
folder_enron5 = '/home/dani/Escritorio/Unizar/IA/TP6s/TP6_filtro_de_spam/' \
                'mails/enron5'
folder_enron6 = '/home/dani/Escritorio/Unizar/IA/TP6s/TP6_filtro_de_spam/' \
                'mails/enron6'
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


# Se entrena el clasificador
alpha = kfold_cross_validation("Multinomial", 5, 1,
                               CountVectorizer(ngram_range=(1, 1))
                               .fit_transform(training_mails + validation_mails),
                               training_labels + validation_labels)

# Loading test data
data6 = load_enron_folder(folder_enron6)
test_mails = data6['test_mails']
test_labels = data6['test_labels']


# Se hace la evaluacion del clasificador
evaluation(alpha,
           (CountVectorizer(ngram_range=(1, 1))
           .fit(training_mails + validation_mails),
            training_mails + validation_mails),
           training_labels + validation_labels,
             test_mails, test_labels, "Multinomial")