# import PyTorch libraries
import os
from threading import Thread
from time import sleep
from click import password_option
from requests import session
from .data_setup import download_and_extract_dataset, download_model_if_missing

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# %pylab inline
import torch
import torchvision
from torchvision import models, datasets  # Parte TL
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import zipfile
# from google.colab import drive
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.nn.modules.activation import ReLU
import torch.nn.functional as F
from datetime import datetime
from torch.nn.modules.activation import ReLU
import torch.nn.functional as F


import pandas as pd
import altair as alt
from altair_saver import save
import numpy as np
from pathlib import Path
import re
from fpdf import FPDF
import sys
#import database as db
import pickle
from pathlib import Path
import re
from PyPDF2 import PdfReader
import io
from PIL import Image
import sys
import base64
import smtplib
from email.message import EmailMessage
import random
import string
import json
import numpy as np
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn import svm
import cv2
import PIL

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor
import dill
import time
import copy


from flask import Blueprint, render_template, request, flash, jsonify, session, redirect, url_for, current_app
from flask_login import login_required, current_user
from flask_dropzone import Dropzone
from .models import User, ModelSaved, ReportPDF
from . import db
#from .kernel import kernel_matrix
import json
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import requests
from jinja2 import Environment
env = Environment(autoescape=True)

views = Blueprint('views', __name__) #views es el name del blueprint



#en este script se registran las rutas a las diferentes páginas de la aplicacion

listaModelsSaved = []
stringPerformance = []
stringPerformanceString = ""
optimizer = None
trainingFinished = False
test_accuracy = None
test_loss = None
current_user_id = None #se usa porque en el thread hijo del training no se puede obtener current_user.id, y así poder disponer de ese valor
show_ModelSavedMessage = False #para saber cuando mostrar el mensaje de model saved en la funcion final updateFlag()
show_pdfReportMessage = False #para saber cuando mostrar el mensaje de pdf report saved en la funcion final updateFlag()
previousGraphImageName = ""
actualGraphImageName = ""
loaded_model = None #almacena el modelo elegido para predecir la imagen en el predictor
choose = None #almacena el nombre del modelo elegido para predecir la imagen en el predictor
randomNumTrainerGraphs = 0
anteriorRandomNumTrainerGraphs = 0
device= None
imageUploaded = False #para saber si no se ha subido ninguna imagen en el drag and drop
counterImages = 0 #para saber si se sube mas de una imagen en el drag and drop
current_time_saveModel = ""
current_time_pdfName = ""
date = ""
path=""

directoryOriginalPath = Path(__file__).parents[1]  # guardo la ruta del proyecto /QuantumKernelspLayDogsVSCats
directoryOriginal = str(directoryOriginalPath)
updateFlag = ""
errorTraining = None

th = Thread()

# ----------------------------------------------------------
# -----------------------CPU OR GPU-------------------------
# ----------------------------------------------------------
# Check to see if we have a GPU to use for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('A {} device was detected.'.format(device))

# Print the name of the cuda device, if detected
if device == 'cuda':
    print(torch.cuda.get_device_name(device=device))


# Funciones alternativas para elegir GPU o CPU, mover los datos a la gpu...
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
#controlz hasta qui

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



# -----------------------------------------------------------------------------------------------------
# -------FUNCIONES PARA LA PREDICCIÓN Y CARGA DEL MODELO QUANTUM KERNEL-------------------
# -----------------------------------------------------------------------------------------------------
n_qubits=4
dev_kernel = qml.device("default.qubit", wires=n_qubits)

projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
    evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])

# -----------------------------------------------------------------------------------------------------
# -------FUNCIONES PARA LA PREDICCIÓN Y CARGA DEL MODELO VARIATIONAL QUANTUM CIRCUIT-------------------
# -----------------------------------------------------------------------------------------------------
n_qubits = 4
q_depth = 6  # Depth of the quantum circuit (number of variational layers)
q_delta = 0.01 # Initial spread of random quantum weights

dev = qml.device("default.qubit", wires=n_qubits) 

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)

class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.pre_net = nn.Linear(4096, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, 2)

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = quantum_net(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)



#-------------------------------------------
#-------------------------------------------
#------FUNCIÓN QUE REALIZA EL ENTRENAMIENTO------
#-------------------------------------------
class ExecuteTrainingTask:
      
    def __init__(self):
        self._running = True
      
    def terminate(self):
        '''
        Función encargada de desactivar el flag running cuando se quiera detener el entrenamiento que está en curso dentro de la función run().
        '''
        self._running = False

    def setRunningTrue(self):
        '''
        Función encargada de activar el flag running a la hora de comenzar el entrenamiento del modelo en la función run().
        '''
        self._running = True
      
    def run(self, datasetTL, workWithTL, modeTL, modelTL, pretrainedTL, svmTL,
                optimizerTL, validationSet_size, lr, num_epochs, batch_size, saveModel, generatePDF, cRegularizationSVM, kernelSVM, gammaSVM, shuffleSVM, typeQuantum):
        '''
        Función encargada de realizar el entrenamiento de un modelo en segundo plano.
        '''
        global trainingFinished
        global test_accuracy
        global test_loss
        global current_user_id
        global show_ModelSavedMessage
        global show_pdfReportMessage
        global previousGraphImageName
        global actualGraphImageName
        global randomNumTrainerGraphs
        global anteriorRandomNumTrainerGraphs
        global device
        global current_time_saveModel
        global current_time_pdfName
        global date
        global path
        global errorTraining
        errorTraining = None
        i=0
        while i < 1:
            try:
                #Download dataset if necessary
                os.chdir(directoryOriginalPath)
                training_dir, test_dir = download_and_extract_dataset(datasetTL, directoryOriginal)
                
                #Download predefined model weights if necessary
                VGG16_MODEL_URL = "https://drive.google.com/file/d/1CxQH-CPjCBYCX9W0Wfp3sDUTBH9voDHS/view?usp=drive_link"
                VGG16_MODEL_PATH = os.path.join("website", "static", "additional", "predictSectionModels", "VGG16_FullyRetrained_definitivo.pth")
                download_model_if_missing(VGG16_MODEL_URL, VGG16_MODEL_PATH)
                VGG16_LL_MODEL_URL = "https://drive.google.com/file/d/1x3UfwsUeT36UMQ0SgkoNcLFlbv2WKpIq/view?usp=drive_link"
                VGG16__LL_MODEL_PATH = os.path.join("website", "static", "additional", "predictSectionModels", "VGG16_LastLayerRetrained_definitivo.pth")
                download_model_if_missing(VGG16_LL_MODEL_URL, VGG16__LL_MODEL_PATH)
                SVM_MODEL_URL = "https://drive.google.com/file/d/1VBNnlvCQ-P1_AB_JEL-lvctCtrIE_FsG/view?usp=drive_link"
                SVM_MODEL_PATH = os.path.join("website", "static", "additional", "predictSectionModels", "SVM_definitivo.pth")
                download_model_if_missing(SVM_MODEL_URL, SVM_MODEL_PATH)
                VGG16_SVM__MODEL_URL = "https://drive.google.com/file/d/1KPbQE32kiLCA1xQc2oBk6KPr49kwVexh/view?usp=drive_link"
                VGG16_SVM_MODEL_PATH = os.path.join("website", "static", "additional", "predictSectionModels", "VGG16_SVM_definitivo.pth")
                download_model_if_missing(VGG16_SVM__MODEL_URL, VGG16_SVM_MODEL_PATH)
                VGG16__QSVM_MODEL_URL = "https://drive.google.com/file/d/1L5zoufNnCUIt2-I0VFm1-2CoTorK1UcO/view?usp=drive_link"
                VGG16__QSVM_MODEL_PATH = os.path.join("website", "static", "additional", "predictSectionModels", "VGG16_QSVM_definitivo.pth")
                download_model_if_missing(VGG16__QSVM_MODEL_URL, VGG16__QSVM_MODEL_PATH)
                VGG16_Var_MODEL_URL = "https://drive.google.com/file/d/1VfEzP-xV__ExNdtq5ryKSOIQH55y2Kee/view?usp=drive_link"
                VGG16_Var_MODEL_PATH = os.path.join("website", "static", "additional", "predictSectionModels", "VGG16_VariationalQuantumCircuit_definitivo.pth")
                download_model_if_missing(VGG16_Var_MODEL_URL, VGG16_Var_MODEL_PATH)
                

                if svmTL == 'YES':

                    if modelTL == 'VGG16':
                        base_model = VGG16(weights='imagenet')
                        model = tf.keras.Model(inputs=base_model.input,      
                        outputs=base_model.get_layer('flatten').output)
                    elif modelTL == 'VGG19':
                        base_model = VGG19(weights='imagenet')
                        model = tf.keras.Model(inputs=base_model.input,      
                        outputs=base_model.get_layer('flatten').output)
                    elif modelTL == 'ResNet50':
                        base_model = ResNet50(weights='imagenet')
                        model = tf.keras.Model(inputs=base_model.input,      
                        outputs=base_model.get_layer('avg_pool').output)
                    elif modelTL == 'MobileNetV2':
                        base_model = MobileNetV2(weights='imagenet')
                        base_model.layers[154]._name = 'capaFinal'
                        model = tf.keras.Model(inputs=base_model.input,      
                        outputs=base_model.get_layer('capaFinal').output)
                    elif modelTL == 'Inception v3':
                        base_model = InceptionV3(weights='imagenet')
                        model = tf.keras.Model(inputs=base_model.input,      
                        outputs=base_model.get_layer('avg_pool').output)
                    elif modelTL == 'DenseNet 121':
                        base_model = DenseNet121(weights='imagenet')
                        model = tf.keras.Model(inputs=base_model.input,      
                        outputs=base_model.get_layer('avg_pool').output)
                    


                    categorias = ['cats', 'dogs']

                    dataset = []

                    def get_features(img_path,nombreModel):
                        if nombreModel != 'Inception v3':
                            img = load_img(img_path, target_size=(224, 224))
                        else:
                            img = load_img(img_path, target_size=(299, 299))
                        #img = cv2.imread(imgpath,1) # el 0 indica que se cargue la imagen en escala de grises, un 1 indica en color
                        #img = cv2.resize( img, (224,224))
                        x = img_to_array(img)
                        x = np.expand_dims(x, axis=0) #la dimension de x era 3 y pasa a ser 4
                        x = preprocess_input(x)
                        flatten = model.predict(x) #la dimension de flatten es 2
                        return list(flatten[0])

                    features, labels = [], []

                    for categoria in categorias:
                        path = os.path.join(training_dir, categoria) #ruta de la carpeta cats o la carpeta dogs
                        label = categorias.index(categoria)
                        i=0
                        for img in os.listdir(path):
                            if i <200:
                                imgpath = os.path.join(path, img) #ruta de la imagen
                                try:
                                    img = PIL.Image.open(imgpath)
                                except PIL.UnidentifiedImageError:
                                        print(imgpath) #imprime el nombre de la imagen si se produce fallo al cargar alguna

                                if (imgpath != training_dir + "\cats\_DS_Store") & (imgpath != training_dir + "\dogs\_DS_Store"): #este archivo esta en la carpeta pero no es una de las imagenes
                                    features.append(get_features(imgpath, modelTL))
                                    labels.append(label)
                                i += 1
                    

                    # Loop into the directory of images and extract features and labels
                    #for image_path in dataset:
                    #    features.append(get_features(image_path, modelTL))
                    #    labels.append(label)
                    
                    if validationSet_size == '10%':
                        val_size = 0.1
                    elif validationSet_size == '15%':
                        val_size = 0.15
                    elif validationSet_size == '20%':
                        val_size = 0.2
                    elif validationSet_size == '25%':
                        val_size = 0.25
                    elif validationSet_size == '30%':
                        val_size = 0.3
                    elif validationSet_size == '35%':
                        val_size = 0.35
                    elif validationSet_size == '40%':
                        val_size = 0.4


                    if shuffleSVM == 'YES':
                        shuffleValue = True
                    elif shuffleSVM == 'NO':
                        shuffleValue = False

                    X_train, X_test, y_train, y_test = train_test_split( features, labels, test_size= val_size, shuffle= shuffleValue)

                    cRegularizationSVMaux = float(cRegularizationSVM)
                    if kernelSVM == 'linear':
                        modelSVM = svm.SVC(C=cRegularizationSVMaux, kernel = kernelSVM, probability=True) #C es la regularización
                    elif gammaSVM != 'auto' and gammaSVM != 'scale':
                        modelSVM = svm.SVC(C=cRegularizationSVMaux, kernel = kernelSVM, gamma = float(gammaSVM), probability=True)
                    else:
                        modelSVM = svm.SVC(C=cRegularizationSVMaux, kernel = kernelSVM, gamma = gammaSVM, probability=True)


                    modelSVM.fit(X_train, y_train) #entrena con las features y labels de los grupos train 

                    predicted = modelSVM.predict(X_test)

                    # get the accuracy
                    test_accuracy = accuracy_score(y_test, predicted)

                    # get the loss
                    test_loss = log_loss(y_test, predicted, labels=labels)
                    


                elif modeTL == 'Quantum':

                    if typeQuantum == 'VGG16 + Quantum Kernel':
                        

                        base_model = VGG16(weights='imagenet')
                        model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
                        modelAux = Sequential()
                        modelAux.add(model)
                        modelAux.add(Conv2D(64, (1,1), activation='relu'))
                        modelAux.add(Conv2D(32, (1,1), activation='relu'))
                        modelAux.add(MaxPooling2D(pool_size=(2, 2)))
                        modelAux.add(Conv2D(16, (1,1), activation='relu'))
                        modelAux.add(Conv2D(8, (1,1), activation='relu'))
                        modelAux.add(Conv2D(4, (1,1), activation='relu'))
                        modelAux.add(MaxPooling2D(pool_size=(2, 2)))
                        modelAux.add(Flatten())

                        categorias = ['cats', 'dogs']

                        dataset = []

                        def get_features(img_path):
                            img = load_img(img_path, target_size=(224, 224))
                            #img = cv2.imread(imgpath,1) # el 0 indica que se cargue la imagen en escala de grises, un 1 indica en color
                            #img = cv2.resize( img, (224,224))
                            x = img_to_array(img)
                            x = np.expand_dims(x, axis=0) #la dimension de x era 3 y pasa a ser 4
                            x = preprocess_input(x)
                            flatten = modelAux.predict(x) #la dimension de flatten es 2
                            print(flatten[0])
                            return list(flatten[0])

                        features, labels = [], []

                        for categoria in categorias:
                            path = os.path.join(training_dir, categoria) #ruta de la carpeta cats o la carpeta dogs
                            label = categorias.index(categoria)
                            i=0
                            for img in os.listdir(path):
                                if i <100:
                                    imgpath = os.path.join(path, img) #ruta de la imagen
                                    try:
                                        img = PIL.Image.open(imgpath)
                                    except PIL.UnidentifiedImageError:
                                            print(imgpath) #imprime el nombre de la imagen si se produce fallo al cargar alguna
                                    if (imgpath != training_dir + "\cats\_DS_Store") & (imgpath != training_dir + "\dogs\_DS_Store"): #este archivo esta en la carpeta pero no es una de las imagenes
                                        features.append(get_features(imgpath))
                                        labels.append(label)
                                i += 1
                        

                        X_train, X_test, y_train, y_test = train_test_split( features, labels, test_size=0.15, shuffle= True)

                        
                        modelQuantumKernel = svm.SVC(kernel=kernel_matrix, probability = True).fit(X_train, y_train)

                        predicted = modelQuantumKernel.predict(X_test) #predice sobre el test set

                        # get the accuracy
                        test_accuracy = accuracy_score(predicted, y_test)

                        # get the loss
                        test_loss = log_loss(y_test, predicted, labels=labels)


                    elif typeQuantum == 'VGG16 + Variational Quantum Circuit':


                        #SETTING OF THE MAIN HYPER-PARAMETERS OF THE MODEL
                        n_qubits = 4                # Number of qubits
                        step = 0.0004               # Learning rate
                        batch_size = 4              # Number of samples for each training step
                        num_epochs = 4              # Number of training epochs
                        q_depth = 6                 # Depth of the quantum circuit (number of variational layers)
                        gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
                        q_delta = 0.01              # Initial spread of random quantum weights
                        start_time = time.time()    # Start of the computation timer

                        dev = qml.device("default.qubit", wires=n_qubits)

                        data_transforms = {
                            "train": transforms.Compose(
                                [
                                    # transforms.RandomResizedCrop(224),     # uncomment for data augmentation
                                    # transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    # Normalize input channels using mean values and standard deviations of ImageNet.
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]
                            ),
                            "test": transforms.Compose(
                                [
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]
                            ),
                        }

                        data_dir = training_dir
                        image_datasets = { 
                        "train":  datasets.ImageFolder(training_dir, data_transforms["train"]),
                        "validation":  datasets.ImageFolder(test_dir, data_transforms["test"])
                        }
                        dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation"]}
                        class_names = image_datasets["train"].classes

                        # Initialize dataloader
                        dataloaders = {
                            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                            for x in ["train", "validation"]
                        }
                    


                        model_hybrid = torchvision.models.vgg16(pretrained=True)

                        for param in model_hybrid.parameters():
                            param.requires_grad = False


                        # Notice that model_hybrid.fc is the last layer of VGG16

                        model_hybrid.classifier[6] = DressedQuantumNet()

                        # Use CUDA or CPU according to the "device" object.
                        model_hybrid = model_hybrid.to(device)

                        criterion = nn.CrossEntropyLoss()
                        optimizer_hybrid = optim.Adam(model_hybrid.classifier[6].parameters(), lr=step)
                        exp_lr_scheduler = lr_scheduler.StepLR(
                            optimizer_hybrid, step_size=10, gamma=gamma_lr_scheduler
                        )

                        def train_model(model, criterion, optimizer, scheduler, num_epochs):
                            since = time.time()
                            best_model_wts = copy.deepcopy(model.state_dict())
                            best_acc = 0.0
                            best_loss = 10000.0  # Large arbitrary number
                            best_acc_train = 0.0
                            best_loss_train = 10000.0  # Large arbitrary number
                            history = []
                            print("Training started:")

                            for epoch in range(num_epochs):

                                # Each epoch has a training and validation phase
                                for phase in ["train", "validation"]:
                                    if phase == "train":
                                        # Set model to training mode
                                        model.train()
                                    else:
                                        # Set model to evaluate mode
                                        model.eval()
                                    running_loss = 0.0
                                    running_corrects = 0

                                    # Iterate over data.
                                    n_batches = dataset_sizes[phase] // batch_size
                                    it = 0
                                    for inputs, labels in dataloaders[phase]:
                                        since_batch = time.time()
                                        batch_size_ = len(inputs)
                                        inputs = inputs.to(device)
                                        labels = labels.to(device)
                                        optimizer.zero_grad()

                                        # Track/compute gradient and make an optimization step only when training
                                        with torch.set_grad_enabled(phase == "train"):
                                            outputs = model(inputs)
                                            _, preds = torch.max(outputs, 1)
                                            loss = criterion(outputs, labels)
                                            if phase == "train":
                                                loss.backward()
                                                optimizer.step()

                                        # Print iteration results
                                        running_loss += loss.item() * batch_size_
                                        batch_corrects = torch.sum(preds == labels.data).item()
                                        running_corrects += batch_corrects
                                        print(
                                            "Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}".format(
                                                phase,
                                                epoch + 1,
                                                num_epochs,
                                                it + 1,
                                                n_batches + 1,
                                                time.time() - since_batch,
                                            ),
                                            end="\r",
                                            flush=True,
                                        )
                                        it += 1

                                    # Print epoch results
                                    epoch_loss = running_loss / dataset_sizes[phase]
                                    epoch_acc = running_corrects / dataset_sizes[phase]
                                    print(
                                        "Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}        ".format(
                                            "train" if phase == "train" else "validation  ",
                                            epoch + 1,
                                            num_epochs,
                                            epoch_loss,
                                            epoch_acc,
                                        )
                                    )

                                    global stringPerformance  #tODO ESTO ES PARA IMPRIMIR DURANTE EL ENTRENAMIENTO COMO LO HACE EN CADA EPOCA
                                    global stringPerformanceString

                                    


                                    if phase == "train":
                                        train_accuracy = epoch_acc
                                        train_loss = epoch_loss
                                    if phase == "validation":
                                        val_accuracy = epoch_acc
                                        val_loss = epoch_loss
                                        result = {'train_loss': train_loss, 'train_acc': train_accuracy, 'val_loss': val_loss, 'val_acc': val_accuracy}
                                        history.append(result)

                                        stringPerformanceAux = f"Epoch [{epoch}]  |  train_loss: {train_loss:.4f}  |  train_acc: {train_accuracy:.4f}  |  val_loss: {val_loss:.4f}  |  val_acc: {val_accuracy:.4f}"
                                        stringPerformance = np.append(stringPerformance, stringPerformanceAux) #para usar en el pdf

                                        numDecimals1 = 4
                                        numDecimals2 = 4 
                                        numDecimals3 = 4 
                                        numDecimals4 = 4
                                        if len(str(int(result['train_loss']))) > 1: #esto es para que muestre siempre 5 cifras y no se descuadren las filas en el Training Started
                                            numDecimals1 = 4 - (len(str(int(result['train_loss']))) - 1)
                                        if len(str(int(result['train_acc']))) > 1:
                                            numDecimals2 = 4 - (len(str(int(result['train_acc']))) - 1)
                                        if len(str(int(result['val_loss']))) > 1:
                                            numDecimals3 = 4 - (len(str(int(result['val_loss']))) - 1)
                                        if len(str(int(result['val_acc']))) > 1:
                                            numDecimals4 = 4 - (len(str(int(result['val_acc']))) - 1)

                                        if numDecimals1 < 0:
                                            train_loss = 10000
                                        if numDecimals2 < 0:
                                            train_accuracy = 10000
                                        if numDecimals3 < 0:
                                            val_loss = 10000
                                        if numDecimals4 < 0:
                                            val_acc = 10000

                                        stringPerformanceAux2 = f"Epoch [{epoch}]  |  train_loss: {train_loss:.{numDecimals1}f}  |  train_acc: {train_accuracy:.{numDecimals2}f}  |  val_loss: {val_loss:.{numDecimals3}f}  |  val_acc: {val_accuracy:.{numDecimals4}f}"
                                                
                                        stringPerformanceString += stringPerformanceAux2 #se usa en la ultima función para mostrar en el html
                                        stringPerformanceString += "\n"



                                    # Check if this is the best model wrt previous epochs
                                    if phase == "validation" and epoch_acc > best_acc:
                                        best_acc = epoch_acc
                                        best_model_wts = copy.deepcopy(model.state_dict())
                                    if phase == "validation" and epoch_loss < best_loss:
                                        best_loss = epoch_loss
                                    if phase == "train" and epoch_acc > best_acc_train:
                                        best_acc_train = epoch_acc
                                    if phase == "train" and epoch_loss < best_loss_train:
                                        best_loss_train = epoch_loss

                                    # Update learning rate
                                    if phase == "train":
                                        scheduler.step()

                            # Print final results
                            model.load_state_dict(best_model_wts)
                            time_elapsed = time.time() - since
                            print(
                                "Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
                            )
                            print("Best test loss: {:.4f} | Best test accuracy: {:.4f}".format(best_loss, best_acc))
                            return (model,best_loss,best_acc, history)


                        model_hybrid, test_loss, test_accuracy, history = train_model(
                            model_hybrid, criterion, optimizer_hybrid, exp_lr_scheduler, num_epochs=num_epochs
                        )

                        def plot_accuracies(history):
                            global actualGraphImageName
                            global randomNumTrainerGraphs
                            global anteriorRandomNumTrainerGraphs
                            # train_accuracies = [x['train_acc'] for x in history]
                            # val_accuracies = [x['val_acc'] for x in history]
                            # x = np.arange(21)
                            data = {
                                'train_accuracies': [x['train_acc'] for x in history],
                                'val_accuracies': [x['val_acc'] for x in history],
                                'epoch': np.arange(num_epochs)
                            }
                            df_acc = pd.DataFrame(data)

                            lines = alt.Chart(df_acc).mark_line().encode(
                                x='epoch',
                                y=alt.Y(alt.repeat("layer"), aggregate="mean", title="Accuracy"),
                                color=alt.datum(alt.repeat("layer"))
                            ).properties(title="Accuracy vs. Nº of epochs").repeat(
                                layer=["train_accuracies", "val_accuracies"])

                            lines.save(directoryOriginal + '/website/static/images/graphAccuracyTrainer.png')


                        def plot_losses(history):
                            data = {
                                'train_losses': [x['train_loss'] for x in history],
                                'val_losses': [x['val_loss'] for x in history],
                                'epoch': np.arange(num_epochs)
                            }
                            df_acc = pd.DataFrame(data)

                            lines = alt.Chart(df_acc).mark_line().encode(
                                x='epoch',
                                y=alt.Y(alt.repeat("layer"), aggregate="mean", title="Losses"),
                                color=alt.datum(alt.repeat("layer"))
                            ).properties(title="Loss vs. Nº of epochs").repeat(layer=["train_losses", "val_losses"])

                            lines.save(directoryOriginal + '/website/static/images/graphLossesTrainer.png')
                            #altair_chart(lines, use_container_width=True)
                        

                        plot_accuracies(history)
                        plot_losses(history)

                else:

                    if (modelTL == "Inception v3"): #INCEPTION V3 REQUIERE IMAGENES DE 299 x 299
                        dataAugmentation_transform = transforms.Compose([
                                # transforms.ToPILImage(),
                                transforms.Resize((299, 299)),
                                # transforms.RandomCrop((30,30)),
                                # transforms.ColorJitter(brightness=0.5),
                                # transforms.RandomResizedCrop(size=(224,224)),
                                transforms.RandomRotation(degrees=45),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomGrayscale(p=0.2),
                                # transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
                                transforms.ToTensor()
                        ])
                    else:
                        dataAugmentation_transform = transforms.Compose([
                                # transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                # transforms.RandomCrop((30,30)),
                                # transforms.ColorJitter(brightness=0.5),
                                # transforms.RandomResizedCrop(size=(224,224)),
                                transforms.RandomRotation(degrees=45),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomGrayscale(p=0.2),
                                # transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
                                transforms.ToTensor()
                        ])


                    dataset = ImageFolder(training_dir, transform=dataAugmentation_transform) #aquí es donde se generan las labels, 0 para gatos y 1 para perros

                    img, label = dataset[0]
                    print(len(dataset))
                    print(img.shape, label)
                    print(dataset.classes)
                    print(dataset)


                    # ----------------------------------------------------------------------------------------------------------------
                    # TRAINING AND VALIDATION DATASETS
                    # ----------------------------------------------------------------------------------------------------------------

                    # Creación del validation set
                    random_seed = 42
                    torch.manual_seed(random_seed)

                    if validationSet_size == '10%':
                        val_size = int(len(dataset) * 0.1)  # aquí se ajusta el tamaño del validation set
                    elif validationSet_size == '15%':
                        val_size = int(len(dataset) * 0.15)
                    elif validationSet_size == '20%':
                        val_size = int(len(dataset) * 0.2)
                    elif validationSet_size == '25%':
                        val_size = int(len(dataset) * 0.25)
                    elif validationSet_size == '30%':
                        val_size = int(len(dataset) * 0.3)
                    elif validationSet_size == '35%':
                        val_size = int(len(dataset) * 0.35)
                    elif validationSet_size == '40%':
                        val_size = int(len(dataset) * 0.4)

                    #val_size = 50 #solamente para hacer las pruebas en poco tiempo
                    train_size = len(dataset) - val_size

                    train_ds, val_ds = random_split(dataset, [train_size, val_size])
                    print("Tamaño del train set: " + str(len(train_ds)) + "\nTamaño del validation set: " + str(len(val_ds)))

                    # Luego el train_ds y el val_ds se separan cada uno de ellos en batches de imagenes:

                    from torch.utils.data.dataloader import DataLoader

                    # batch_size = 32  # aquí se ajusta el tamaño de los batch, se suele ir doblando 64, 128, 256...

                    # creacion del train dataloader y validation dataloader que crean los batches
                    train_dl = DataLoader(train_ds,
                                            int(batch_size),
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=True)
                    val_dl = DataLoader(val_ds,
                                        int(batch_size * 2),
                                        num_workers=0,
                                        pin_memory=True)  # duplicamos el batch_size para el validation dataloader porque no vamos usar
                                                        # gradiente para la validation por lo que solo necesitaremos la mitad de la memoria

                    # ----------------------------------------------------------------------------------------------------------------
                    # DEFINING THE MODEL (CNN)
                    # ----------------------------------------------------------------------------------------------------------------

                    # función que realiza la operación de convolution
                    def apply_kernel(image, kernel):
                        ri, ci = image.shape  # image dimensions
                        rk, ck = kernel.shape  # kernel dimensions
                        ro, co = ri - rk + 1, ci - ck + 1  # output dimensions
                        output = torch.zeros([ro, co])
                        for i in range(ro):
                            for j in range(co):
                                output[i, j] = torch.sum(image[i:i + rk, j:j + ck] * kernel)
                        return output


                    if pretrainedTL == 'YES':
                        pretrainedValue = True
                    elif pretrainedTL == 'NO':
                        pretrainedValue = False

                    # Cargar el modelo VGG16
                    if (modelTL == "VGG16"):
                        modelSelected = models.vgg16(pretrained = pretrainedValue)
                    elif (modelTL == "VGG19"):
                        modelSelected = models.vgg19(pretrained = pretrainedValue)
                    elif (modelTL == "ResNet18"):
                        modelSelected = models.resnet18(pretrained = pretrainedValue)
                    elif (modelTL == "ResNet50"):
                        modelSelected = models.resnet50(pretrained = pretrainedValue)
                    elif (modelTL == "MobileNetV2"):
                        modelSelected = models.mobilenet_v2(pretrained = pretrainedValue)
                    elif (modelTL == "AlexNet"):
                        modelSelected = models.alexnet(pretrained = pretrainedValue)
                    elif (modelTL == "GoogLeNet"):
                        modelSelected = models.googlenet(pretrained = pretrainedValue)
                    elif (modelTL == "Inception v3"):
                        modelSelected = models.inception_v3(pretrained = pretrainedValue, aux_logits=False)
                    elif (modelTL == "SqueezeNet 1_0"):
                        modelSelected = models.squeezenet1_0(pretrained = pretrainedValue)
                    elif (modelTL == "DenseNet 121"):
                        modelSelected = models.densenet121(pretrained = pretrainedValue)
                    

                    if pretrainedValue == True:
                        for i, parameter in enumerate(modelSelected.parameters()):
                            parameter.requires_grad = False


                    if (modelTL == "VGG16") | (modelTL == "VGG19") | (modelTL == "AlexNet"):
                        modelSelected.classifier[6] = nn.Sequential(
                            nn.Linear(4096, 512),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(512, 2)
                        )  # SE MODIFICA EL CLASIFICADOR PARA QUE CLASIFIQUE SOLAMENTE 2 CLASES
                    elif (modelTL == "ResNet18") | (modelTL == "ResNet50") | (modelTL == "GoogLeNet"):
                        num_ftrs = modelSelected.fc.in_features
                        modelSelected.fc = nn.Linear(num_ftrs, 2)
                    elif (modelTL == "Inception v3"):
                        num_ftrs = modelSelected.fc.in_features
                        #modelSelected.AuxLogits.fc = nn.Linear(768, 2)
                        modelSelected.fc = nn.Linear(num_ftrs, 2)
                    elif (modelTL == "MobileNetV2"):
                        num_ftrs = modelSelected.classifier[1].in_features
                        modelSelected.classifier[1] = nn.Linear(num_ftrs, 2, bias=True)
                    elif (modelTL == "SqueezeNet 1_0"):
                        final_conv = nn.Conv2d(512, 2, kernel_size=1) #2 clases a clasificar
                        modelSelected.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
                    elif (modelTL == "DenseNet 121"):
                        num_ftrs = modelSelected.classifier.in_features
                        modelSelected.classifier = nn.Linear(num_ftrs, 2, bias=True)



                    if (modelTL == "VGG16") | (modelTL == "VGG19") | (modelTL == "MobileNetV2") | (modelTL == "AlexNet") | (modelTL == "SqueezeNet 1_0") | (modelTL == "DenseNet 121"):
                        for i, parameter in enumerate(modelSelected.classifier.parameters()):
                            parameter.requires_grad = True  # Se ponen a TRUE solo las capas del Classifier para que SÍ se entrenen
                    elif (modelTL == "ResNet18") | (modelTL == "ResNet50") | (modelTL == "GoogLeNet") | (modelTL == "Inception v3"):
                        for i, parameter in enumerate(modelSelected.fc.parameters()):
                            parameter.requires_grad = True  # Se ponen a TRUE solo las capas de la ultima fully conected layer para que SÍ se entrenen


                    

                    # Primero definimos un modelo base llamado ImageClassificationBase que contiene métodos (funciones) de ayuda
                    # para el training y validation, y que son comunmente usadas.
                    class ImageClassificationBase(nn.Module):
                        def training_step(self,batch):  # self representa el objeto que se va a ir creando eventualmente (sería como el propio modelo)
                            images, labels = batch
                            out = self(images)  # Generate predictions, se pasa el batch de images al modelo(self)
                            loss = F.cross_entropy(out, labels)  # Calculate loss
                            acc = accuracy(out, labels)  # Calculate accuracy
                            # return loss
                            return {'train_loss': loss, 'train_acc': acc}

                        def validation_step(self, batch):
                            images, labels = batch
                            out = self(images)  # Generate predictions, se pasa el batch de images al modelo(self)
                            loss = F.cross_entropy(out, labels)  # Calculate loss
                            acc = accuracy(out, labels)  # Calculate accuracy
                            return {'val_loss': loss.detach(),
                                    'val_acc': acc}  # devuelve la perdida de validation y la precisión de validation             #COMENTADO PARA IMPRIMIR TRAIN_ACC EN TB

                        def validation_epoch_end(self, outputs):  # toma las perdidas y precisiones de todos los diferentes batches del validation data y los combina calculando su media y devuelve una unica perdida y precisión para todo el validation set
                            batch_losses = [x['val_loss'] for x in outputs]
                            epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
                            batch_accs = [x['val_acc'] for x in outputs]
                            epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
                            return {'val_loss': epoch_loss.item(),
                                    'val_acc': epoch_acc.item()}  # COMENTADO PARA IMPRIMIR TRAIN_ACC EN TB

                        def epoch_end(self, epoch, result):  # toma los resultados del epoch y los muestra

                            print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                                epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))
                            #st.write(
                            #    "Epoch [{}]  |  train_loss: {:.4f}  |  train_acc: {:.4f}  |  val_loss: {:.4f}  |  val_acc: {:.4f}".format(
                            #        epoch, result['train_loss'], result['train_acc'], result['val_loss'],
                            #        result['val_acc']))  # Imprime el número de epocas y el accuracy y loss para cada época

                            global stringPerformance
                            global stringPerformanceString

                            stringPerformanceAux = "Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f} \n".format(
                                epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc'])
                            stringPerformance = np.append(stringPerformance, stringPerformanceAux) #para usar en el pdf

                            numDecimals1 = 4
                            numDecimals2 = 4 
                            numDecimals3 = 4 
                            numDecimals4 = 4
                            if len(str(int(result['train_loss']))) > 1: #esto es para que muestre siempre 5 cifras y no se descuadren las filas en el Training Started
                                numDecimals1 = 4 - (len(str(int(result['train_loss']))) - 1)
                            if len(str(int(result['train_acc']))) > 1:
                                numDecimals2 = 4 - (len(str(int(result['train_acc']))) - 1)
                            if len(str(int(result['val_loss']))) > 1:
                                numDecimals3 = 4 - (len(str(int(result['val_loss']))) - 1)
                            if len(str(int(result['val_acc']))) > 1:
                                numDecimals4 = 4 - (len(str(int(result['val_acc']))) - 1)

                            
                            train_loss = result['train_loss']
                            train_accuracy = result['train_acc']
                            val_loss = result['val_loss']
                            val_accuracy = result['val_acc']

                            if numDecimals1 < 0:
                                train_loss = 10000
                            if numDecimals2 < 0:
                                train_accuracy = 10000
                            if numDecimals3 < 0:
                                val_loss = 10000
                            if numDecimals4 < 0:
                                val_accuracy = 10000
                            

                            stringPerformanceAux2 = f"Epoch [{epoch}]  |  train_loss: {train_loss:.{numDecimals1}f}  |  train_acc: {train_accuracy:.{numDecimals2}f}  |  val_loss: {val_loss:.{numDecimals3}f}  |  val_acc: {val_accuracy:.{numDecimals4}f}"
                        
                            stringPerformanceString += stringPerformanceAux2 #se usa en la ultima función para mostrar en el html
                            stringPerformanceString += "\n"

                            # pdf.cell(50, 8, "Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                            #    epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']), ln = True)


                    def accuracy(outputs, labels):  # esta funcion calcula la precision de la prediccion
                        _, preds = torch.max(outputs, dim=1)
                        return torch.tensor(torch.sum(preds == labels).item() / len(
                            preds))  # devuelve la etiqueta(label) que más aparece y la compara con las verdaderas etiquetas



                    # Ahora creamos nuestro propio modelo que extiende el ImageClassificationBase
                    class CatsVSDogsCnnModel(ImageClassificationBase):
                        def __init__(self):
                            super().__init__()
                            self.network = modelSelected

                        def forward(self, xb):
                            return self.network(xb)


                    model = CatsVSDogsCnnModel()  # SE CREA EL MODELO
                    # print(model)

                    for images, labels in train_dl:
                        print('images.shape:', images.shape)
                        out = model(images)
                        print('out.shape:', out.shape)
                        print('out[0]:', out[0])
                        break

                    # ----------------------------------------------------------------------------------------------------------------
                    # AQUÍ IRIAN LAS FUNCIONES ALTERNATIVAS PARA ELEGIR GPU O CPU, MOVER LOS DATOS A LA GPU...
                    # ----------------------------------------------------------------------------------------------------------------
                    
                    



                    # ----------------------------------------------------------------------------------------------------------------
                    # TRAINING THE MODEL
                    # ----------------------------------------------------------------------------------------------------------------
                    @torch.no_grad()  # indica que mientras se ejecute la funcion evaluate no se compute ningún gradiente
                    def evaluate(model, val_loader):
                        model.eval()  # informa a pytorch de que estamos evaluando el modelo por lo que no habrá randomize
                        outputs = [model.validation_step(batch) for batch in
                                    val_loader]  # obtiene batches de imagenes del val_dl y los pasa a la validation_step function que devolvera la loss de la validation
                        return model.validation_epoch_end(outputs)  # calcula la media de las loss y devuelve un unico output


                    def fit(epochs, lr, model, train_loader, val_loader,
                            opt_func):  # se le pasa el numero de epochs y el optimizador que usaremos SGD (stocastic gradient descend).
                        history = []
                        global optimizer
                        optimizer = opt_func(model.parameters(),lr)  # el optimizador toma los model.parameters que son los weights y biases de todas las capas y los va actualizando
                        for epoch in range(epochs):  # para cada epoch va a ver una fase de training y otra de validation
                            # Training Phase

                            if self._running == False:
                                break


                            model.train()  # informa a pytorch de que estamos entrenando el modelo
                            train_losses = []  # se mantiene un seguimiento de las perdidas(losses)
                            train_acc = []  # se mantiene un seguimiento de las perdidas(losses)
                            for batch in train_loader:  # cogemos batches de imagenes del train_dl

                                if self._running == False:
                                    break

                                loss = model.training_step(batch)  # esta funcion esta definida en ImageClassificationBase class y devuelve la perdida(loss) para el batch que se le pasa como input
                                train_losses.append(loss['train_loss'])  # para obtener al final la loss total del epoch
                                train_acc.append(loss['train_acc'])  # para obtener al final la loss total del epoch
                                loss['train_loss'].backward()  # calcula los gradientes
                                optimizer.step()  # se aplica el decenso de gradiente con un optimizador (actualiza los parametros)
                                optimizer.zero_grad()  # pone a 0 los gradientes calculados en loss.backwards
                            # Validation phase
                            result = evaluate(model,val_loader)  # llama a la funcion evaluate definida más arriba en este bloque y devuelve el validation loss y validation accuracy
                            result['train_loss'] = torch.stack(train_losses).mean().item()  # calcula la media del train_losses para el epoch entero
                            result['train_acc'] = torch.stack(train_acc).mean().item()  # calcula la media del train_acc para el epoch entero
                            model.epoch_end(epoch,result)  # imprime el numero de epoch, el training loss, validation loss y validation accuracy
                            history.append(result)  # el resultado se añade al registro de resultados anteriores
                        
                        return history


                    # -------------------------------------------------------------------------------
                    # -----------------------EVALUATE ON VALIDATION DATASET -----------------------------------
                    # -------------------------------------------------------------------------------
                    evaluate(model, val_dl)

                    if (optimizerTL == 'SGD'):
                        opt_func = torch.optim.SGD  # usamos la función de optimizacion SGD
                    elif (optimizerTL == 'Adam'):
                        opt_func = torch.optim.Adam  # usamos la función de optimizacion Adam
                    elif (optimizerTL == 'Adadelta'):
                        opt_func = torch.optim.Adadelta  # usamos la función de optimizacion Adadelta
                    elif (optimizerTL == 'Adagrad'):
                        opt_func = torch.optim.Adagrad  # usamos la función de optimizacion Adagrad
                    elif (optimizerTL == 'AdamW'):
                        opt_func = torch.optim.AdamW  # usamos la función de optimizacion AdamW
                    elif (optimizerTL == 'NAdam'):
                        opt_func = torch.optim.NAdam  # usamos la función de optimizacion NAdam
                    elif (optimizerTL == 'RAdam'):
                        opt_func = torch.optim.RAdam  # usamos la función de optimizacion RAdam
                    elif (optimizerTL == 'Rprop'):
                        opt_func = torch.optim.Rprop  # usamos la función de optimizacion Rprop
                    elif (optimizerTL == 'ASGD'):
                        opt_func = torch.optim.ASGD  # usamos la función de optimizacion ASGD

                    
                    history = fit(num_epochs, lr, model, train_dl, val_dl,opt_func) #entrenamos
                    if self._running == False:
                        break

                    def plot_accuracies(history):
                            global actualGraphImageName
                            global randomNumTrainerGraphs
                            global anteriorRandomNumTrainerGraphs
                            # train_accuracies = [x['train_acc'] for x in history]
                            # val_accuracies = [x['val_acc'] for x in history]
                            # x = np.arange(21)
                            data = {
                                'train_accuracies': [x['train_acc'] for x in history],
                                'val_accuracies': [x['val_acc'] for x in history],
                                'epoch': np.arange(num_epochs)
                            }
                            df_acc = pd.DataFrame(data)

                            lines = alt.Chart(df_acc).mark_line().encode(
                                x='epoch',
                                y=alt.Y(alt.repeat("layer"), aggregate="mean", title="Accuracy"),
                                color=alt.datum(alt.repeat("layer"))
                            ).properties(title="Accuracy vs. Nº of epochs").repeat(
                                layer=["train_accuracies", "val_accuracies"])

                            #save(str(lines), directoryOriginal + '/website/static/images/graphAccuracyTrainer.png')
                            lines.save(directoryOriginal + '/website/static/images/graphAccuracyTrainer.png')

                            #nowGrapAccuracy = datetime.now()
                            #current_time = nowGrapAccuracy.strftime("grapAccuracy_%d_%m_%Y_%H_%M.png")
                            #actualGraphImageName = current_time

                            #if(os.path.isfile(directoryOriginal + '/website/static/images/graphAccuracyTrainer' + str(anteriorRandomNumTrainerGraphs) + '.png')):
                            #    os.remove(directoryOriginal + '/website/static/images/graphAccuracyTrainer' + str(anteriorRandomNumTrainerGraphs) + '.png')

                            #save(lines, directoryOriginal + '/website/static/images/' + actualGraphImageName)
                            #save(lines, directoryOriginal + '/website/static/images/graphAccuracyTrainer' + str(randomNumTrainerGraphs) + '.png')
                            #altair_chart(lines, use_container_width=True)


                    def plot_losses(history):
                        data = {
                            'train_losses': [x['train_loss'] for x in history],
                            'val_losses': [x['val_loss'] for x in history],
                            'epoch': np.arange(num_epochs)
                        }
                        df_acc = pd.DataFrame(data)

                        lines = alt.Chart(df_acc).mark_line().encode(
                            x='epoch',
                            y=alt.Y(alt.repeat("layer"), aggregate="mean", title="Losses"),
                            color=alt.datum(alt.repeat("layer"))
                        ).properties(title="Loss vs. Nº of epochs").repeat(layer=["train_losses", "val_losses"])

                        #save(str(lines), directoryOriginal + '/website/static/images/graphLossesTrainer.png')
                        lines.save(directoryOriginal + '/website/static/images/graphLossesTrainer.png')
                        #altair_chart(lines, use_container_width=True)
                    

                    plot_accuracies(history)
                    plot_losses(history)



                    # -------------------------------------------------------------------------------
                    # -----------------------EVALUATE ON TEST DATASET -----------------------------------
                    # -------------------------------------------------------------------------------
                    transform = transforms.Compose(
                        [transforms.Resize((224, 224)),
                            transforms.ToTensor()])  # transforma todas las imagenes en el mismo tamaño de 224x224 pixeles

                    test_dataset = ImageFolder(test_dir, transform=transform)  # creamos un dataset con la clase ImageFolder

                    test_loader = DeviceDataLoader(DataLoader(test_dataset, int(batch_size * 2)), device)
                    result = evaluate(model, test_loader)
                    test_accuracy = result['val_acc'] #test accuracy para pasarla al html y mostrarla
                    test_loss = result['val_loss'] #test loss para pasarla al html y mostrarla



            except Exception as e: #PARA CAPTURAR LAS EXPECEPCIONES Y PODER MOSTRAR MENSAJE DE ERROR EN TRAININGS STARTED QUE INDIQUE QUE HACER
                print('Exception occurred while code execution: ' + repr(e))
                errorTraining = str(repr(e))




            # -------------------------------------------------------------------------------
            # -----------------------GUARDAR MODELO Y DESCARGAR PDF -------------------------
            # -------------------------------------------------------------------------------
            now = datetime.now()
            directoryAdditional = directoryOriginal + '/website/static/additional'

            current_time_saveModel = now.strftime("model_%d_%m_%Y_%H_%M.pth")
            current_time_pdfName = now.strftime("pdfModel_%d_%m_%Y_%H_%M.pdf")
            date = now.strftime("%d/%m/%Y %H:%M:%S")

            if saveModel is not None:
                
                if self._running == False:
                        break

                #control z hasta aqui

                if svmTL == 'YES':
                    state = {
                        'type': 'svm',
                        'c': cRegularizationSVM,
                        'kernel': kernelSVM,
                        'gamma': gammaSVM,
                        'modelTL': modelTL,
                        'test_accuracy': test_accuracy,
                        'model_state': modelSVM,
                        'datasetTL': datasetTL,
                        'validationSet_size': validationSet_size
                    }

                elif modeTL == 'Quantum':

                    if typeQuantum == 'VGG16 + Variational Quantum Circuit':

                        state = {
                            'type': 'quantum',
                            'typeQuantum': 'VariationalQuantumCircuit',
                            'model_state': model_hybrid.state_dict(),
                            'test_accuracy': test_accuracy,
                            'history': history,
                            'epoch': num_epochs,
                            'datasetTL': datasetTL,
                            'shuffleSVM': shuffleSVM,
                            'validationSet_size': validationSet_size,
                            'kernelSVM': kernelSVM,
                            'gammaSVM': gammaSVM

                        }

                    else:
                        state = {
                        'type': 'quantum',
                        'typeQuantum': 'QuantumKernel',
                        'test_accuracy': test_accuracy,
                        'model_state': modelQuantumKernel,
                        'datasetTL': datasetTL
                    }

                else:
                    state = {
                        'type': 'normal',
                        'epoch': num_epochs,
                        'lr': lr,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'history': history,
                        'modelTL': modelTL,
                        'pretrained': pretrainedValue,
                        'test_accuracy': test_accuracy,
                        'datasetTL': datasetTL,
                        'pretrainedTL': pretrainedTL,
                        'optimizerTL': optimizerTL,
                        'validationSet_size': validationSet_size,
                        'batch_size': batch_size

                    }
                

                
                path= ""
                if not os.path.exists(directoryAdditional + '/modelsSaved'):
                    os.makedirs(directoryAdditional + '/modelsSaved')
                if current_user_id == 1:
                    if not os.path.exists(directoryAdditional + '/modelsSaved/admin'):
                        os.makedirs(directoryAdditional + '/modelsSaved/admin')
                    torch.save(state, directoryAdditional + '/modelsSaved/admin/' + current_time_saveModel) #si de fallo aqui es por falta de memoria RAM
                    path= directoryAdditional + '/modelsSaved/admin/' + current_time_saveModel
                else:
                    if not os.path.exists(directoryAdditional + '/modelsSaved/user' + str(current_user_id)):
                        os.makedirs(directoryAdditional + '/modelsSaved/user' + str(current_user_id))
                    torch.save(state, directoryAdditional + '/modelsSaved/user' + str(current_user_id) + '/' + current_time_saveModel) #si de fallo aqui es por falta de memoria RAM
                    path= directoryAdditional + '/modelsSaved/user' + str(current_user_id) + '/' + current_time_saveModel
                
                show_ModelSavedMessage = True
                #el mensaje de model saved lo muestro en la función final updateFlag()
            


            # ----------------------------------------------------------
            # ----------------------- PARA EL PDF -------------------------
            # ----------------------------------------------------------
            class PDF(FPDF):
                def header(self):
                    self.image(directoryOriginal + '/website/static/images/logoAPP2.PNG', 10, 8, 15) #logo
                    self.set_font('helvetica', 'B', 20) #font
                    self.cell(15)
                    self.cell(0,15, 'RESULTS OBTAINED FOR THE CHOSEN MODEL', border = False, ln = True, align = 'C') #titulo
                    #self.cell(0, 15, '', border=False, ln=True)
                    self.ln(15)

                def footer(self):
                    self.set_y(-15) # que empiece 15 por encima del final
                    self.set_font('helvetica', 'I', 10)
                    self.cell(0,10, f'Page {self.page_no()}/{{nb}}', align='C') #poner el nº de pagina


            if generatePDF is not None:

                
                if self._running == False:
                        break
                
                if svmTL == 'YES':
                    #PDF PARA MODELOS SVM
                    pdf = PDF('P', 'mm', 'Letter')  # crea el objeto pdf
                    pdf.alias_nb_pages()  # para obtener el numero total de páginas
                    pdf.set_auto_page_break(auto=True, margin=15)  # fija que se cambie de página automáticamente
                    pdf.add_page()  # añade una página
                    pdf.set_font('times', '', 12)  # especifica la letra

                    current_time = now.strftime("Date: %d/%m/%Y %H:%M:%S")
                    pdf.set_font('helvetica', 'BI', 16)
                    pdf.cell(0, 15, current_time, border=False, ln=True, align='L')
                    pdf.ln(5)
                    pdf.set_font('helvetica', 'BU', 16)
                    pdf.cell(0, 15, 'CHOSEN CHARACTERISTICS', border=False, ln=True, align='L')
                    pdf.set_font('times', '', 12)
                    pdf.cell(0, 15, 'Dataset: ' + datasetTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Work with: ' + workWithTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Mode: ' + 'Classical' + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Model for Transfer Learning: ' + modelTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Import the model as pretrained?: ' + 'YES' + '\n', border=False, ln=True,align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Use SVM as the last layer?: ' + 'YES' + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'C (Regularization): ' + str(cRegularizationSVM) + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Kernel: ' + kernelSVM + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    if kernelSVM != 'linear':
                        pdf.cell(0, 15, 'Gamma: ' + str(gammaSVM) + '\n', border=False, ln=True, align='L')
                        pdf.ln(1)
                    pdf.cell(0, 15, 'Size of the validation set with respect to the training set: ' + str(validationSet_size) + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Shuffle dataset when splitting it: ' + shuffleSVM + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    if saveModel == 'on':
                        pdf.cell(0, 15, 'Save the model: ' + 'YES' + '\n', border=False, ln=True, align='L')
                    else:
                        pdf.cell(0, 15, 'Save the model: ' + 'NO' + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    if generatePDF == 'on':
                        pdf.cell(0, 15, 'Generate PDF report: ' + 'YES' + '\n', border=False, ln=True, align='L')
                    else:
                        pdf.cell(0, 15, 'Generate PDF report: ' + 'NO' + '\n', border=False, ln=True, align='L')
                    pdf.ln(5)

                    pdf.set_font('helvetica', 'BU', 16)
                    pdf.cell(0, 15, 'MODEL PERFORMANCE', border=False, ln=True, align='L')
                    pdf.set_font('times', '', 12)
                    pdf.cell(0, 15, 'Test Accuracy: {:.3f}'.format(test_accuracy) + '\n', border=False, ln=True,
                                align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Test Loss: {:.3f}'.format(test_loss) + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)

                elif modeTL == 'Quantum':
                    #PDF PARA MODELOS QUANTUM
                    
                    #if typeQuantum == 'VGG16 + Variational Quantum Circuit':
                    pdf = PDF('P', 'mm', 'Letter')  # crea el objeto pdf
                    pdf.alias_nb_pages()  # para obtener el numero total de páginas
                    pdf.set_auto_page_break(auto=True, margin=15)  # fija que se cambie de página automáticamente
                    pdf.add_page()  # añade una página
                    pdf.set_font('times', '', 12)  # especifica la letra

                    current_time = now.strftime("Date: %d/%m/%Y %H:%M:%S")
                    pdf.set_font('helvetica', 'BI', 16)
                    pdf.cell(0, 15, current_time, border=False, ln=True, align='L')
                    pdf.ln(5)
                    pdf.set_font('helvetica', 'BU', 16)
                    pdf.cell(0, 15, 'CHOSEN CHARACTERISTICS', border=False, ln=True, align='L')
                    pdf.set_font('times', '', 12)
                    pdf.cell(0, 15, 'Dataset: ' + datasetTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Work with: ' + workWithTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Mode: ' + 'Quantum' + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Type of quantum model: ' + typeQuantum + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    if saveModel == 'on':
                        pdf.cell(0, 15, 'Save the model: ' + 'YES' + '\n', border=False, ln=True, align='L')
                    else:
                        pdf.cell(0, 15, 'Save the model: ' + 'NO' + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    if generatePDF == 'on':
                        pdf.cell(0, 15, 'Generate PDF report: ' + 'YES' + '\n', border=False, ln=True, align='L')
                    else:
                        pdf.cell(0, 15, 'Generate PDF report: ' + 'NO' + '\n', border=False, ln=True, align='L')
                    pdf.ln(5)

                    if typeQuantum == 'VGG16 + Variational Quantum Circuit':
                        pdf.set_font('helvetica', 'BU', 16)
                        pdf.cell(0, 15, 'TRAINING', border=False, ln=True, align='L')
                        pdf.set_font('times', '', 12)
                        for performance in stringPerformance:
                            pdf.cell(0, 8, performance, border=False, ln=True, align='L')
                            pdf.ln(1)
                        
                        pdf.add_page()  # añade una página
                        pdf.set_font('helvetica', 'BU', 16)
                        pdf.cell(0, 15, 'RESULTS', border=False, ln=True, align='C')
                        pdf.image(directoryOriginal + '/website/static/images/graphAccuracyTrainer.png', x=40, w=150)
                        pdf.ln(5)
                        pdf.image(directoryOriginal + '/website/static/images/graphLossesTrainer.png', x=40, w=150)
                        
                        pdf.ln(5)

                    pdf.set_font('helvetica', 'BU', 16)
                    pdf.cell(0, 15, 'MODEL PERFORMANCE', border=False, ln=True, align='L')
                    pdf.set_font('times', '', 12)
                    pdf.cell(0, 15, 'Test Accuracy: {:.3f}'.format(test_accuracy) + '\n', border=False, ln=True,
                                align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Test Loss: {:.3f}'.format(test_loss) + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)

                else:
                    #PDF PARA MODELOS NORMALES

                    pdf = PDF('P', 'mm', 'Letter')  # crea el objeto pdf
                    pdf.alias_nb_pages()  # para obtener el numero total de páginas
                    pdf.set_auto_page_break(auto=True, margin=15)  # fija que se cambie de página automáticamente
                    pdf.add_page()  # añade una página
                    pdf.set_font('times', '', 12)  # especifica la letra

                    current_time = now.strftime("Date: %d/%m/%Y %H:%M:%S")
                    pdf.set_font('helvetica', 'BI', 16)
                    pdf.cell(0, 15, current_time, border=False, ln=True, align='L')
                    pdf.ln(5)
                    pdf.set_font('helvetica', 'BU', 16)
                    pdf.cell(0, 15, 'CHOSEN CHARACTERISTICS', border=False, ln=True, align='L')
                    pdf.set_font('times', '', 12)
                    pdf.cell(0, 15, 'Dataset: ' + datasetTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Work with: ' + workWithTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Mode: ' + modeTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Model for Transfer Learning: ' + modelTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Import the model as pretrained?: ' + pretrainedTL + '\n', border=False, ln=True,align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Use SVM as the last layer?: ' + svmTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Optimizer: ' + optimizerTL + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Learning rate: ' + str(lr) + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Number of epochs: ' + str(num_epochs) + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Size of the validation set with respect to the training set: ' + str(validationSet_size) + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Batch Size: ' + str(batch_size) + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    if saveModel == 'on':
                        pdf.cell(0, 15, 'Save the model: ' + 'YES' + '\n', border=False, ln=True, align='L')
                    else:
                        pdf.cell(0, 15, 'Save the model: ' + 'NO' + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)
                    if generatePDF == 'on':
                        pdf.cell(0, 15, 'Generate PDF report: ' + 'YES' + '\n', border=False, ln=True, align='L')
                    else:
                        pdf.cell(0, 15, 'Generate PDF report: ' + 'NO' + '\n', border=False, ln=True, align='L')
                    pdf.ln(5)
                    pdf.set_font('helvetica', 'BU', 16)
                    pdf.cell(0, 15, 'TRAINING', border=False, ln=True, align='L')
                    pdf.set_font('times', '', 12)
                    for performance in stringPerformance:
                        pdf.cell(0, 8, performance, border=False, ln=True, align='L')
                        pdf.ln(1)
                    # pdf.ln(5)

                    pdf.add_page()  # añade una página
                    pdf.set_font('helvetica', 'BU', 16)
                    pdf.cell(0, 15, 'RESULTS', border=False, ln=True, align='C')
                    pdf.image(directoryOriginal + '/website/static/images/graphAccuracyTrainer.png', x=40, w=150)
                    pdf.ln(5)
                    pdf.image(directoryOriginal + '/website/static/images/graphLossesTrainer.png', x=40, w=150)

                    pdf.add_page()  # añade una página
                    pdf.set_font('helvetica', 'BU', 16)
                    pdf.cell(0, 15, 'MODEL PERFORMANCE', border=False, ln=True, align='L')
                    pdf.set_font('times', '', 12)
                    pdf.cell(0, 15, 'Test Accuracy: {:.3f}'.format(result['val_acc']) + '\n', border=False, ln=True,
                                align='L')
                    pdf.ln(1)
                    pdf.cell(0, 15, 'Test Loss: {:.3f}'.format(result['val_loss']) + '\n', border=False, ln=True, align='L')
                    pdf.ln(1)


                path = ""
                if not os.path.exists(directoryAdditional + '/pdfReports'):
                    os.makedirs(directoryAdditional + '/pdfReports')
                if current_user_id == 1:
                    if not os.path.exists(directoryAdditional + '/pdfReports/admin'):
                        os.makedirs(directoryAdditional + '/pdfReports/admin')
                    pdf.output(directoryAdditional + '/pdfReports/admin/' + current_time_pdfName)  # fija el nombre del pdf
                    path = directoryAdditional + '/pdfReports/admin/' + current_time_pdfName
                else:
                    if not os.path.exists(directoryAdditional + '/pdfReports/user' + str(current_user_id)):
                        os.makedirs(directoryAdditional + '/pdfReports/user' + str(current_user_id))
                    pdf.output(directoryAdditional + '/pdfReports/user' + str(current_user_id) + '/' + current_time_pdfName)  # fija el nombre del pdf
                    path = directoryAdditional + '/pdfReports/user' + str(current_user_id) + '/' + current_time_pdfName

                show_pdfReportMessage = True
                #el mensaje de pdf report saved lo muestro en la función final updateFlag()               

            i = i + 1
        
        self._running = True #lo pongo a true para que la siguiente vez que de a "Train model" este a true y entre
        #session['endTraining'] = True
        trainingFinished = True
        



c = ExecuteTrainingTask()




#----------------------------
#-------- HOME ------------
#----------------------------

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    '''
    La función home() carga la vista Home.
    '''
    stringPerformanceString = "" #para borrar la string de la performance del trainer cuando se sale del trainer

    return render_template("home.html", user=current_user)


#CONTROL Z HASTA AQUI

# ----------------------------------------------------------
# ----------------------- PREDICTOR -------------------------
# ----------------------------------------------------------

@views.route('/predictor', methods=['GET', 'POST'])
@login_required
def predictor():
    '''
    La función predictor() se encarga de realizar la predicción de una imagen usando el modelo elegido por el usuario.
    '''

    global c
    c.terminate()

    global loaded_model
    global choose
    global imageUploaded
    global counterImages
    classPredicted = None
    catProbability = None
    dogProbability = None
    #session['test_accuracy_loaded'] = None
    #loaded_model = None #para poner a none el modelo si todavia no se ha elegido y no se ha pulsado el boton Use this model

    stringPerformanceString = "" #para borrar la string de la performance del trainer cuando se sale del trainer

    if current_user.username == 'admin':
        pathUserModels = directoryOriginal + '/website/static/additional/modelsSaved/admin'
    else:
        pathUserModels = directoryOriginal + '/website/static/additional/modelsSaved/user' + str(current_user.id)
    print(str(pathUserModels))

    session["modelsSaved"]= None
    session["existsModelsSaved"] = True
    session["buttonUseThisModel"]= False
    session["buttonPredict"]= False
    session['test_accuracy_loaded'] = None
    session['optimizerTL_loaded'] = None
    session['num_epochs_loaded'] = None
    session['lr_loaded'] = None
    session['pretrainedTL_loaded'] = None
    session['datasetTL_loaded'] = None
    session['validationSet_size_loaded'] = None
    session['batch_size_loaded'] = None
    session['kernel_loaded'] = None
    session['gamma_loaded'] = None
    session['c_loaded'] = None
    if 'tipo' not in session:
        session['tipo'] = 'normal'
    if 'typeQuantum' not in session:
        session['typeQuantum'] = 'VariationalQuantumCircuit'
    if 'modelTL_loaded' not in session:
        session['modelTL_loaded'] = None
    
    

    if not os.path.exists(pathUserModels):
        session["existsModelsSaved"] = False
    else:
        arrayModelos = os.listdir(pathUserModels)
        numModelos = len(arrayModelos)
        if numModelos == 0:
            session["existsModelsSaved"] = False
        else:
            session["modelsSaved"] = arrayModelos
    
    if request.method == 'POST':

        botonPulsado = request.form['botonesPredictPage']

        if botonPulsado == 'botonPredictImage': #SI SE PULSA EL BOTON PREDICT
            if imageUploaded == False:
                session["buttonUseThisModel"]= True
                flash('You must upload one image in order to predict it!', category='error')
            elif (counterImages < 1) or (counterImages > 1):
                session["buttonUseThisModel"]= True
                flash('You must upload only one image in order to predict it!', category='error')
            else:
                session["buttonUseThisModel"]= True
                session["buttonPredict"]= True

                imagePath = directoryOriginal + '/website/static/images/predictionImage.jpg'
                image = Image.open(imagePath)


                if choose == "SVM":

                    imagen = cv2.imread(imagePath,1) # el 0 indica que se cargue la imagen en escala de grises, un 1 indica en color
                    imagen = cv2.resize( imagen, (128,128))
                    imagen = np.array(imagen).flatten() 
                    #imagenToPredict.append() 

                    dataset = []
                    dataset.append([imagen, 'label'])
                    features = []
                    labels = []

                    for feature,label in dataset:
                        features.append(feature)
                    
                    print(choose)
                    prediccion = loaded_model.predict_proba(features)

                    count = 0
                    for i in prediccion[0]:
                        if count == 0:
                            catProbability = round(i*100, 2) #almacena la probabilidad de que sea gato
                        else:
                            dogProbability = round(i*100, 2) #almacena la probabilidad de que sea perro
                        count += 1
                    
                    if catProbability > dogProbability:
                        classPredicted = "Cat"
                    else:
                        classPredicted = "Dog"

                elif choose == "VGG16 + SVM":

                    base_model = VGG16(weights='imagenet')
                    model = tf.keras.Model(inputs=base_model.input,      
                    outputs=base_model.get_layer('flatten').output)

                    def get_features(img_path):
                        img = load_img(img_path, target_size=(224, 224))
                        #img = cv2.imread(imgpath,1) # el 0 indica que se cargue la imagen en escala de grises, un 1 indica en color
                        #img = cv2.resize( img, (224,224))
                        x = img_to_array(img)
                        x = np.expand_dims(x, axis=0) #la dimension de x era 3 y pasa a ser 4
                        x = preprocess_input(x)
                        flatten = model.predict(x) #la dimension de flatten es 2
                        return list(flatten[0])


                    imagenToPredict=[]

                    imagenToPredict.append(get_features(imagePath)) #le debo aplicar a la imagen el mismo preprocesado que a las que se usaron para entrenar

                    prediccion = loaded_model.predict_proba(imagenToPredict)

                    count = 0
                    for i in prediccion[0]:
                        if count == 0:
                            catProbability = round(i*100, 2) #almacena la probabilidad de que sea gato
                        else:
                            dogProbability = round(i*100, 2) #almacena la probabilidad de que sea perro
                        count += 1
                    
                    if catProbability > dogProbability:
                        classPredicted = "Cat"
                    else:
                        classPredicted = "Dog"


                elif choose == "VGG16 + Quantum Kernel":

                    base_model = VGG16(weights='imagenet')
                    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
                    modelAux = Sequential()
                    modelAux.add(model)
                    modelAux.add(Conv2D(64, (1,1), activation='relu'))
                    modelAux.add(Conv2D(32, (1,1), activation='relu'))
                    modelAux.add(MaxPooling2D(pool_size=(2, 2)))
                    modelAux.add(Conv2D(16, (1,1), activation='relu'))
                    modelAux.add(Conv2D(8, (1,1), activation='relu'))
                    modelAux.add(Conv2D(4, (1,1), activation='relu'))
                    modelAux.add(MaxPooling2D(pool_size=(2, 2)))
                    modelAux.add(Flatten())

                    def get_features(img_path):
                        img = load_img(img_path, target_size=(224, 224))
                        #img = cv2.imread(imgpath,1) # el 0 indica que se cargue la imagen en escala de grises, un 1 indica en color
                        #img = cv2.resize( img, (224,224))
                        x = img_to_array(img)
                        x = np.expand_dims(x, axis=0) #la dimension de x era 3 y pasa a ser 4
                        x = preprocess_input(x)
                        flatten = modelAux.predict(x) #la dimension de flatten es 2
                        print(flatten[0])
                        return list(flatten[0])
                    

                    imagenToPredict=[]

                    print("features de la imagen")
                    print(imagenToPredict.append(get_features(imagePath))) #le debo aplicar a la imagen el mismo preprocesado que a las que se usaron para entrenar

                    
                    prediccion = loaded_model.predict_proba(imagenToPredict)

                    count = 0
                    for i in prediccion[0]:
                        if count == 0:
                            catProbability = round(i*100, 2) #almacena la probabilidad de que sea gato
                        else:
                            dogProbability = round(i*100, 2) #almacena la probabilidad de que sea perro
                        count += 1
                    
                    if catProbability > dogProbability:
                        classPredicted = "Cat"
                    else:
                        classPredicted = "Dog"


                elif choose == "VGG16 + Variational Quantum Circuit":

                    def predict_image(img, model):
                        # Convert to a batch of 1
                        # xb = to_device(img.unsqueeze(0), device)
                        # Get predictions from model
                        xb = img.unsqueeze(0)
                        print(xb)
                        print(xb.shape)

                        yb = model(xb)
                        print(yb)
                        print(yb.shape)

                        sm = torch.nn.Softmax()
                        probabilities = sm(yb)  #USAMOS SOFTMAX PARA OBTENER LAS PROBABILIDADES DE SER PERRO O GATO, en probabilities[0] está la probabilidad de ser gato y en probabilities[1] la de ser perro
                        #st.write(probabilities)

                        # SE ESCOGE COMO ELEGIDA LA CLASE CON MAYOR PROBABILIDAD
                        _, preds = torch.max(yb, dim=1)
                        print(preds[0].item()) #preds[0].item() vale 0 (gato) o 1 (perro)

                        classDogOrCat = preds[0].item()
                        if classDogOrCat == 0:
                            valueReturn = "Cat"
                        elif classDogOrCat == 1:
                            valueReturn = "Dog"

                        resultados = (valueReturn, probabilities[0][0], probabilities[0][1])

                        # Retrieve the class label
                        return resultados

                    transform2 = transforms.Compose(
                        [transforms.Resize((224, 224)),
                        transforms.ToTensor()])

                    uploaded_img = transform2(image)
                    img = uploaded_img

                    resultados = predict_image(img, loaded_model)
                    classPredicted = resultados[0]
                    catProbability = float(resultados[1]*100)
                    catProbability = round(catProbability, 2)
                    dogProbability = float(resultados[2] * 100)
                    dogProbability = round(dogProbability, 2)

                else: #si se ha elegido VGG17 Fully Retrained o Last Layer Retrained, o un modelo propio
                    
                    
                    modelTL_loaded = session['modelTL_loaded']
                    if session['tipo'] == 'svm':
                        if modelTL_loaded == 'VGG16':
                            base_model = VGG16(weights='imagenet')
                            model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
                        elif modelTL_loaded == 'VGG19':
                            base_model = VGG19(weights='imagenet')
                            model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
                        elif modelTL_loaded == 'ResNet50':
                            base_model = ResNet50(weights='imagenet')
                            model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
                        elif modelTL_loaded == 'MobileNetV2':
                            base_model = MobileNetV2(weights='imagenet')
                            base_model.layers[154]._name = 'capaFinal'
                            model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('capaFinal').output)
                        elif modelTL_loaded == 'Inception v3':
                            base_model = InceptionV3(weights='imagenet')
                            model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
                        elif modelTL_loaded == 'DenseNet 121':
                            base_model = DenseNet121(weights='imagenet')
                            model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
                


                        def get_features(img_path,nombreModel):
                            if nombreModel != 'Inception v3':
                                img = load_img(img_path, target_size=(224, 224))
                            else:
                                img = load_img(img_path, target_size=(299, 299))
                            #img = cv2.imread(imgpath,1) # el 0 indica que se cargue la imagen en escala de grises, un 1 indica en color
                            #img = cv2.resize( img, (224,224))
                            x = img_to_array(img)
                            x = np.expand_dims(x, axis=0) #la dimension de x era 3 y pasa a ser 4
                            x = preprocess_input(x)
                            flatten = model.predict(x) #la dimension de flatten es 2
                            return list(flatten[0])

                        imagenToPredict=[]

                        imagenToPredict.append(get_features(imagePath, modelTL_loaded)) #le debo aplicar a la imagen el mismo preprocesado que a las que se usaron para entrenar

                        prediccion = loaded_model.predict_proba(imagenToPredict)

                        count = 0
                        for i in prediccion[0]:
                            if count == 0:
                                catProbability = round(i*100, 2) #almacena la probabilidad de que sea gato
                            else:
                                dogProbability = round(i*100, 2) #almacena la probabilidad de que sea perro
                            count += 1
                        
                        if catProbability > dogProbability:
                            classPredicted = "Cat"
                        else:
                            classPredicted = "Dog"



                    elif session['tipo'] == 'quantum':
                        
                        if session['typeQuantum'] == "VariationalQuantumCircuit": #SI EL MODELO PROPIO ES DE TIPO VARIATIONAL QUANTUM CIRCUIT
                            def predict_image(img, model):
                                # Convert to a batch of 1
                                # xb = to_device(img.unsqueeze(0), device)
                                # Get predictions from model
                                xb = img.unsqueeze(0)
                                print(xb)
                                print(xb.shape)

                                yb = model(xb)
                                print(yb)
                                print(yb.shape)

                                sm = torch.nn.Softmax()
                                probabilities = sm(yb)  #USAMOS SOFTMAX PARA OBTENER LAS PROBABILIDADES DE SER PERRO O GATO, en probabilities[0] está la probabilidad de ser gato y en probabilities[1] la de ser perro
                                #st.write(probabilities)

                                # SE ESCOGE COMO ELEGIDA LA CLASE CON MAYOR PROBABILIDAD
                                _, preds = torch.max(yb, dim=1)
                                print(preds[0].item()) #preds[0].item() vale 0 (gato) o 1 (perro)

                                classDogOrCat = preds[0].item()
                                if classDogOrCat == 0:
                                    valueReturn = "Cat"
                                elif classDogOrCat == 1:
                                    valueReturn = "Dog"

                                resultados = (valueReturn, probabilities[0][0], probabilities[0][1])

                                # Retrieve the class label
                                return resultados

                            transform2 = transforms.Compose(
                                [transforms.Resize((224, 224)),
                                transforms.ToTensor()])

                            uploaded_img = transform2(image)
                            img = uploaded_img

                            resultados = predict_image(img, loaded_model)
                            classPredicted = resultados[0]
                            catProbability = float(resultados[1]*100)
                            catProbability = round(catProbability, 2)
                            dogProbability = float(resultados[2] * 100)
                            dogProbability = round(dogProbability, 2)
                        

                        else: #SI EL MODELO PROPIO ES DE TIPO QUANTUM KERNEL

                            base_model = VGG16(weights='imagenet')
                            model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
                            modelAux = Sequential()
                            modelAux.add(model)
                            modelAux.add(Conv2D(64, (1,1), activation='relu'))
                            modelAux.add(Conv2D(32, (1,1), activation='relu'))
                            modelAux.add(MaxPooling2D(pool_size=(2, 2)))
                            modelAux.add(Conv2D(16, (1,1), activation='relu'))
                            modelAux.add(Conv2D(8, (1,1), activation='relu'))
                            modelAux.add(Conv2D(4, (1,1), activation='relu'))
                            modelAux.add(MaxPooling2D(pool_size=(2, 2)))
                            modelAux.add(Flatten())

                            def get_features(img_path):
                                img = load_img(img_path, target_size=(224, 224))
                                #img = cv2.imread(imgpath,1) # el 0 indica que se cargue la imagen en escala de grises, un 1 indica en color
                                #img = cv2.resize( img, (224,224))
                                x = img_to_array(img)
                                x = np.expand_dims(x, axis=0) #la dimension de x era 3 y pasa a ser 4
                                x = preprocess_input(x)
                                flatten = modelAux.predict(x) #la dimension de flatten es 2
                                print(flatten[0])
                                return list(flatten[0])
                            

                            imagenToPredict=[]

                            print("features de la imagen")
                            print(imagenToPredict.append(get_features(imagePath))) #le debo aplicar a la imagen el mismo preprocesado que a las que se usaron para entrenar

                            
                            prediccion = loaded_model.predict_proba(imagenToPredict)

                            count = 0
                            for i in prediccion[0]:
                                if count == 0:
                                    catProbability = round(i*100, 2) #almacena la probabilidad de que sea gato
                                else:
                                    dogProbability = round(i*100, 2) #almacena la probabilidad de que sea perro
                                count += 1
                            
                            if catProbability > dogProbability:
                                classPredicted = "Cat"
                            else:
                                classPredicted = "Dog"


                    else: #SI EL MODELO PROPIO ES DE TIPO NORMAL
                        
                        def predict_image(img, model):
                            # Convert to a batch of 1
                            # xb = to_device(img.unsqueeze(0), device)
                            # Get predictions from model
                            xb = img.unsqueeze(0)
                            print(xb)
                            print(xb.shape)

                            yb = model(xb)
                            print(yb)
                            print(yb.shape)

                            sm = torch.nn.Softmax()
                            probabilities = sm(yb)  #USAMOS SOFTMAX PARA OBTENER LAS PROBABILIDADES DE SER PERRO O GATO, en probabilities[0] está la probabilidad de ser gato y en probabilities[1] la de ser perro
                            #st.write(probabilities)

                            # SE ESCOGE COMO ELEGIDA LA CLASE CON MAYOR PROBABILIDAD
                            _, preds = torch.max(yb, dim=1)
                            print(preds[0].item()) #preds[0].item() vale 0 (gato) o 1 (perro)

                            classDogOrCat = preds[0].item()
                            if classDogOrCat == 0:
                                valueReturn = "Cat"
                            elif classDogOrCat == 1:
                                valueReturn = "Dog"

                            resultados = (valueReturn, probabilities[0][0], probabilities[0][1])

                            # Retrieve the class label
                            return resultados

                        transform2 = transforms.Compose(
                            [transforms.Resize((224, 224)),
                            transforms.ToTensor()])

                        uploaded_img = transform2(image)
                        img = uploaded_img

                        resultados = predict_image(img, loaded_model)
                        classPredicted = resultados[0]
                        catProbability = float(resultados[1]*100)
                        catProbability = round(catProbability, 2)
                        dogProbability = float(resultados[2] * 100)
                        dogProbability = round(dogProbability, 2)


                flash('Image predicted!', category='success')
                imageUploaded = False
                counterImages = 0

            return render_template("predictor.html", user=current_user, modelsSaved = session["modelsSaved"], existsModelsSaved=session["existsModelsSaved"],
                            buttonUseThisModel=session["buttonUseThisModel"],buttonPredict=session["buttonPredict"],
                             classPredicted=classPredicted,
                            catProbability=catProbability, dogProbability=dogProbability, modelChosen=choose, test_accuracy_loaded=session['test_accuracy_loaded'] , tipo=session['tipo'], typeQuantum=session['typeQuantum'],
                            modelTL_loaded=session['modelTL_loaded'], optimizerTL_loaded=session['optimizerTL_loaded'], num_epochs_loaded=session['num_epochs_loaded'], lr_loaded=session['lr_loaded'],
                            pretrainedTL_loaded=session['pretrainedTL_loaded'], datasetTL_loaded=session['datasetTL_loaded'], validationSet_size_loaded=session['validationSet_size_loaded'],
                            batch_size_loaded=session['batch_size_loaded'], kernel_loaded=session['kernel_loaded'], gamma_loaded=session['gamma_loaded'], c_loaded=session['c_loaded'])
                

        else: #SI SE PULSA EL BOTON USE THIS MODEL
            session["buttonUseThisModel"]= True
            flash('Model ready to be used!', category='success')

            tipoModelo = request.form.get("selectBox") #vale predefinedModels o MyModels
            choose = request.form["RadioOptionsPredefinedModels"]
            print(str(tipoModelo))
            print(str(choose))


            class ImageClassificationBase(nn.Module):
                def training_step(self,
                                    batch):  # self representa el objeto que se va a ir creando eventualmente (sería como el propio modelo)
                    images, labels = batch
                    out = self(images)  # Generate predictions, se pasa el batch de images al modelo(self)
                    loss = F.cross_entropy(out, labels)  # Calculate loss
                    acc = accuracy(out, labels)  # Calculate accuracy
                    # return loss
                    return {'train_loss': loss, 'train_acc': acc}

                def validation_step(self, batch):
                    images, labels = batch
                    out = self(images)  # Generate predictions, se pasa el batch de images al modelo(self)
                    loss = F.cross_entropy(out, labels)  # Calculate loss
                    acc = accuracy(out, labels)  # Calculate accuracy
                    return {'val_loss': loss.detach(),
                            'val_acc': acc}  # devuelve la perdida de validation y la precisión de validation             #COMENTADO PARA IMPRIMIR TRAIN_ACC EN TB

                def validation_epoch_end(self,
                                            outputs):  # toma las perdidas y precisiones de todos los diferentes batches del validation data y los combina calculando su media y devuelve una unica perdida y precisión para todo el validation set
                    batch_losses = [x['val_loss'] for x in outputs]
                    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
                    batch_accs = [x['val_acc'] for x in outputs]
                    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
                    return {'val_loss': epoch_loss.item(),
                            'val_acc': epoch_acc.item()}  # COMENTADO PARA IMPRIMIR TRAIN_ACC EN TB

                def epoch_end(self, epoch, result):  # toma los resultados del epoch y los muestra
                    print(
                        "Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                            epoch, result['train_loss'], result['train_acc'], result['val_loss'],
                            result['val_acc']))
        
                    global stringPerformance
                    stringPerformanceAux = "Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f} \n".format(
                        epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc'])
                    stringPerformance = np.append(stringPerformance, stringPerformanceAux)


            def accuracy(outputs, labels):  # esta funcion calcula la precision de la prediccion
                _, preds = torch.max(outputs, dim=1)
                return torch.tensor(torch.sum(preds == labels).item() / len(
                    preds))  # devuelve la etiqueta(label) que más aparece y la compara con las verdaderas etiquetas


            # Ahora creamos nuestro propio modelo que extiende el ImageClassificationBase
            class CatsVSDogsCnnModel(ImageClassificationBase):
                def __init__(self):
                    super().__init__()
                    self.network = modelSelected

                def forward(self, xb):
                    return self.network(xb)

            pretrained_value = True
            arquitecturaModelo = "VGG16"
            session['tipo'] = 'normal'
            device = torch.device('cpu')
            #loaded_state = torch.load(directoryOriginal + '/website/static/additional/predictSectionModels/VGG16_FeatureExtraction.pth',map_location=device)
            if choose == "VGG16 Fully Retrained":
                loaded_state = torch.load(directoryOriginal + '/website/static/additional/predictSectionModels/VGG16_FullyRetrained_definitivo.pth',map_location=device)
                session['tipo'] = 'normal'
            elif choose == "VGG16 Last Layer Retrained":
                loaded_state = torch.load(directoryOriginal + '/website/static/additional/predictSectionModels/VGG16_LastLayerRetrained_definitivo.pth',map_location=device)
                session['tipo'] = 'normal'
            elif choose == "SVM":
                loaded_state = torch.load(directoryOriginal + '/website/static/additional/predictSectionModels/SVM_definitivo.pth',map_location=device)
                loaded_model = svm.SVC(C=0.5, kernel = 'poly', gamma = 'auto', probability=True) #creo un svm
                loaded_model = loaded_state #lo igualo al svm que teniamos guardado en el archivo .pth
                session['tipo'] = 'svm'
            elif choose == "VGG16 + SVM":
                loaded_state = torch.load(directoryOriginal + '/website/static/additional/predictSectionModels/VGG16_SVM_definitivo.pth',map_location=device)
                loaded_model = svm.SVC(C=0.5, kernel = 'poly', gamma = 'auto', probability=True)
                loaded_model = loaded_state
                session['tipo'] = 'svm'
            elif choose == "VGG16 + Quantum Kernel": 

                loaded_state = torch.load(directoryOriginal + '/website/static/additional/predictSectionModels/VGG16_QSVM_definitivo.pth',map_location=device, pickle_module=dill)
                #loaded_state = torch.load(directoryOriginal + '/website/static/additional/predictSectionModels/VGG16_QSVM_definitivo.pth',map_location=device, pickle_module=dill)#necesario guardarlo y cargarlo con pickle=dill
                
                loaded_model = svm.SVC(kernel=kernel_matrix, probability = True)
                loaded_model = loaded_state['model_state']
                session['test_accuracy_loaded'] = loaded_state['test_accuracy']
                
                session['tipo'] = 'quantum'
                
                


            elif choose == "VGG16 + Variational Quantum Circuit":

 
                model_hybrid = torchvision.models.vgg16(pretrained=True)

                for param in model_hybrid.parameters():
                    param.requires_grad = False

                # Notice that model_hybrid.fc is the last layer of VGG16

                model_hybrid.classifier[6] = DressedQuantumNet()

                # Use CUDA or CPU according to the "device" object.
                loaded_model = model_hybrid.to(device)              #ATENTO QUE PUEDE QUE DE FALLO AQUI AL CAMBIAR DE NOMBRE

                loaded_state = torch.load(directoryOriginal + '/website/static/additional/predictSectionModels/VGG16_VariationalQuantumCircuit_definitivo.pth',map_location=device)

                loaded_history = loaded_state['history']
                num_epochs = loaded_state['num_epochs']
                    
                def plot_accuracies(history):
                    global actualGraphImageName
                    global randomNumTrainerGraphs
                    global anteriorRandomNumTrainerGraphs
                    # train_accuracies = [x['train_acc'] for x in history]
                    # val_accuracies = [x['val_acc'] for x in history]
                    # x = np.arange(21)
                    data = {
                        'train_accuracies': [x['train_acc'] for x in history],
                        'val_accuracies': [x['val_acc'] for x in history],
                        'epoch': np.arange(num_epochs)
                    }
                    df_acc = pd.DataFrame(data)

                    lines = alt.Chart(df_acc).mark_line().encode(
                        x='epoch',
                        y=alt.Y(alt.repeat("layer"), aggregate="mean", title="Accuracy"),
                        color=alt.datum(alt.repeat("layer"))
                    ).properties(title="Accuracy vs. Nº of epochs").repeat(
                        layer=["train_accuracies", "val_accuracies"])

                    lines.save(directoryOriginal + '/website/static/images/graphAccuracy.png')


                def plot_losses(history):
                    data = {
                        'train_losses': [x['train_loss'] for x in history],
                        'val_losses': [x['val_loss'] for x in history],
                        'epoch': np.arange(num_epochs)
                    }
                    df_acc = pd.DataFrame(data)

                    lines = alt.Chart(df_acc).mark_line().encode(
                        x='epoch',
                        y=alt.Y(alt.repeat("layer"), aggregate="mean", title="Losses"),
                        color=alt.datum(alt.repeat("layer"))
                    ).properties(title="Loss vs. Nº of epochs").repeat(layer=["train_losses", "val_losses"])

                    lines.save(directoryOriginal + '/website/static/images/graphLosses.png')
                    #altair_chart(lines, use_container_width=True)
                

                plot_accuracies(loaded_history)
                plot_losses(loaded_history)


                loaded_model.load_state_dict(loaded_state['model']) #cargo el modelo
                session['tipo'] = 'quantum'

            elif choose is not None:
                loaded_state = torch.load(pathUserModels + '/' + choose, map_location=device) #carga el modelo propio del usuario
                session['tipo'] = loaded_state['type']

                if session['tipo'] == 'svm': #SI ES UN MODELO PROPIO PERO QUE FUE ENTRENADO DE TIPO SVM
                    c_loaded = loaded_state['c']
                    kernel_loaded = loaded_state['kernel']
                    gamma_loaded = loaded_state['gamma']
                    modelTL_loaded = loaded_state['modelTL']
                    session['test_accuracy_loaded'] = round(loaded_state['test_accuracy'],2)
                    session['validationSet_size_loaded'] = loaded_state['validationSet_size']
                    session['kernel_loaded'] = loaded_state['kernel']
                    session['gamma_loaded'] = loaded_state['gamma']
                    session['modelTL_loaded'] = loaded_state['modelTL']
                    session['datasetTL_loaded'] = loaded_state['datasetTL']
                    session['c_loaded'] = loaded_state['c']
                    print(modelTL_loaded)
                    print(session['modelTL_loaded'])

                    if kernel_loaded == 'linear':
                        loaded_model = svm.SVC(C=float(c_loaded), kernel = kernel_loaded, probability=True) #creo un svm
                    elif gamma_loaded != 'auto' and gamma_loaded != 'scale':
                        loaded_model = svm.SVC(C=float(c_loaded), kernel = kernel_loaded, gamma = float(gamma_loaded), probability=True)
                    else:
                        loaded_model = svm.SVC(C=float(c_loaded), kernel = kernel_loaded, gamma = gamma_loaded, probability=True)

                    loaded_model = loaded_state['model_state']


                elif session['tipo'] == 'quantum': #SI ES UN MODELO PROPIO PERO QUE FUE ENTRENADO DE TIPO QUANTUM

                    tipoDeModeloQuantumGuardado = loaded_state['typeQuantum']
                    session['typeQuantum'] = tipoDeModeloQuantumGuardado
                    session['test_accuracy_loaded'] = round(loaded_state['test_accuracy'],2)
                    session['datasetTL_loaded'] = loaded_state['datasetTL']

                    if tipoDeModeloQuantumGuardado == 'VariationalQuantumCircuit':
                        
                        model_hybrid = torchvision.models.vgg16(pretrained=True)

                        for param in model_hybrid.parameters():
                            param.requires_grad = False

                        # Notice that model_hybrid.fc is the last layer of VGG16

                        model_hybrid.classifier[6] = DressedQuantumNet()

                        # Use CUDA or CPU according to the "device" object.
                        loaded_model = model_hybrid
                        loaded_model = loaded_model.to(device)              #ATENTO QUE PUEDE QUE DE FALLO AQUI AL CAMBIAR DE NOMBRE

                        #loaded_state = torch.load(directoryOriginal + '/website/static/additional/predictSectionModels/VGG16_VariationalQuantumCircuit_definitivo.pth',map_location=device)

                        loaded_history = loaded_state['history']
                        num_epochs = loaded_state['epoch']

                        def plot_accuracies(history):
                            global actualGraphImageName
                            global randomNumTrainerGraphs
                            global anteriorRandomNumTrainerGraphs
                            # train_accuracies = [x['train_acc'] for x in history]
                            # val_accuracies = [x['val_acc'] for x in history]
                            # x = np.arange(21)
                            data = {
                                'train_accuracies': [x['train_acc'] for x in history],
                                'val_accuracies': [x['val_acc'] for x in history],
                                'epoch': np.arange(num_epochs)
                            }
                            df_acc = pd.DataFrame(data)

                            lines = alt.Chart(df_acc).mark_line().encode(
                                x='epoch',
                                y=alt.Y(alt.repeat("layer"), aggregate="mean", title="Accuracy"),
                                color=alt.datum(alt.repeat("layer"))
                            ).properties(title="Accuracy vs. Nº of epochs").repeat(
                                layer=["train_accuracies", "val_accuracies"])

                            lines.save(directoryOriginal + '/website/static/images/graphAccuracy.png')


                        def plot_losses(history):
                            data = {
                                'train_losses': [x['train_loss'] for x in history],
                                'val_losses': [x['val_loss'] for x in history],
                                'epoch': np.arange(num_epochs)
                            }
                            df_acc = pd.DataFrame(data)

                            lines = alt.Chart(df_acc).mark_line().encode(
                                x='epoch',
                                y=alt.Y(alt.repeat("layer"), aggregate="mean", title="Losses"),
                                color=alt.datum(alt.repeat("layer"))
                            ).properties(title="Loss vs. Nº of epochs").repeat(layer=["train_losses", "val_losses"])

                            lines.save(directoryOriginal + '/website/static/images/graphLosses.png')
                            #altair_chart(lines, use_container_width=True)
                        

                        plot_accuracies(loaded_history)
                        plot_losses(loaded_history)

                        loaded_model.load_state_dict(loaded_state['model_state']) #cargo el modelo
                    
                    else:
                        
                        loaded_model = svm.SVC(kernel=kernel_matrix, probability = True)
                        loaded_model = loaded_state['model_state']
                        session['test_accuracy_loaded'] = round(loaded_state['test_accuracy'],2)


                    session['tipo'] = 'quantum'


                else: #SI ES UN MODELO PROPIO PERO QUE FUE ENTRENADO DE TIPO NORMAL
                    arquitecturaModelo = loaded_state['modelTL']
                    pretrained_value = loaded_state['pretrained']
                    session['modelTL_loaded'] = loaded_state['modelTL']
                    session['test_accuracy_loaded'] = round(loaded_state['test_accuracy'],2)
                    session['optimizerTL_loaded'] = loaded_state['optimizerTL']
                    session['num_epochs_loaded'] = loaded_state['epoch']
                    session['lr_loaded'] = loaded_state['lr']
                    session['pretrainedTL_loaded'] = loaded_state['pretrainedTL']
                    session['datasetTL_loaded'] = loaded_state['datasetTL']
                    session['validationSet_size_loaded'] = loaded_state['validationSet_size']
                    session['batch_size_loaded'] = loaded_state['batch_size']

            

            if (choose != "SVM") & (choose != "VGG16 + SVM") & (session['tipo'] != "svm") & (session['tipo'] != "quantum"):
                print("entro aquiiiii")
                if choose is not None:
                    if (choose == "VGG16 Fully Retrained") | (choose == "VGG16 Last Layer Retrained") | (choose == "VGG16 + SVM") | (choose == "VGG16 + Quantum Kernel") | (choose == "VGG16 + Variational Quantum Circuit"):
                        modelSelected = models.vgg16(True)
                    elif (arquitecturaModelo == "VGG16"):
                        modelSelected = models.vgg16(pretrained=pretrained_value)
                    elif (arquitecturaModelo == "VGG19"):
                        modelSelected = models.vgg19(pretrained=pretrained_value)
                    elif (arquitecturaModelo == "ResNet18"):
                        modelSelected = models.resnet18(pretrained=pretrained_value)
                    elif (arquitecturaModelo == "ResNet50"):
                        modelSelected = models.resnet50(pretrained=pretrained_value)
                    elif (arquitecturaModelo == "MobileNetV2"):
                        modelSelected = models.mobilenet_v2(pretrained=pretrained_value)
                    elif (arquitecturaModelo == "AlexNet"):
                        modelSelected = models.alexnet(pretrained=pretrained_value)
                    elif (arquitecturaModelo == "GoogLeNet"):
                        modelSelected = models.googlenet(pretrained=pretrained_value)
                    elif (arquitecturaModelo == "Inception v3"):
                        modelSelected = models.inception_v3(pretrained=pretrained_value)
                    elif (arquitecturaModelo == "SqueezeNet 1_0"):
                        modelSelected = models.squeezenet1_0(pretrained=pretrained_value)
                    elif (arquitecturaModelo == "DenseNet 121"):
                        modelSelected = models.densenet121(pretrained=pretrained_value)
                    
                    if (pretrained_value == True):
                        for i, parameter in enumerate(modelSelected.parameters()):
                            parameter.requires_grad = False

                    if (arquitecturaModelo == "VGG16") | (arquitecturaModelo == "VGG19") | (arquitecturaModelo == "AlexNet") | (choose == "VGG16 Fully Retrained") | (choose == "VGG16 Last Layer Retrained") | (choose == "VGG16 + SVM") | (choose == "VGG16 + QUANTUM"):
                        modelSelected.classifier[6] = nn.Sequential(
                            nn.Linear(4096, 512),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(512, 2)
                        )  # SE MODIFICA EL CLASIFICADOR PARA QUE CLASIFIQUE SOLAMENTE 2 CLASES
                    elif (arquitecturaModelo == "ResNet18") | (arquitecturaModelo == "ResNet50") | (arquitecturaModelo == "GoogLeNet") | (arquitecturaModelo == "Inception v3"):
                        num_ftrs = modelSelected.fc.in_features
                        modelSelected.fc = nn.Linear(num_ftrs, 2)
                    elif (arquitecturaModelo == "MobileNetV2"):
                        num_ftrs = modelSelected.classifier[1].in_features
                        modelSelected.classifier[1] = nn.Linear(num_ftrs, 2, bias=True)
                    elif (arquitecturaModelo == "SqueezeNet 1_0"):
                        final_conv = nn.Conv2d(512, 2, kernel_size=1) #2 clases a clasificar
                        modelSelected.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
                    elif (arquitecturaModelo == "DenseNet 121"):
                        num_ftrs = modelSelected.classifier.in_features
                        modelSelected.classifier = nn.Linear(num_ftrs, 2, bias=True)


                    if (arquitecturaModelo == "VGG16") | (arquitecturaModelo == "VGG19") | (arquitecturaModelo == "MobileNetV2") | (arquitecturaModelo == "AlexNet") | (arquitecturaModelo == "SqueezeNet 1_0") | (arquitecturaModelo == "DenseNet 121") | (choose == "VGG16 Fully Retrained") | (choose == "VGG16 Last Layer Retrained") | (choose == "VGG16 + SVM") | (choose == "VGG16 + QUANTUM"):
                        for i, parameter in enumerate(modelSelected.classifier.parameters()):
                            parameter.requires_grad = True  # Se ponen a TRUE solo las capas del Classifier para que SÍ se entrenen
                    elif (arquitecturaModelo == "ResNet18") | (arquitecturaModelo == "ResNet50") | (arquitecturaModelo == "GoogLeNet") | (arquitecturaModelo == "Inception v3"):
                        for i, parameter in enumerate(modelSelected.fc.parameters()):
                            parameter.requires_grad = True  # Se ponen a TRUE solo las capas de la ultima fully conected layer para que SÍ se entrenen


                    loaded_model = CatsVSDogsCnnModel()  # SE CREA EL MODELO
                    
                    loaded_model.load_state_dict(loaded_state["model_state"])  # carga el model_state en el nuevo modelo
                    loaded_history = loaded_state['history']
                    loaded_num_epochs = loaded_state['epoch']


                    def plot_accuracies(history):
                        # train_accuracies = [x['train_acc'] for x in history]
                        # val_accuracies = [x['val_acc'] for x in history]
                        # x = np.arange(21)
                        data = {
                            'train_accuracies': [x['train_acc'] for x in history],
                            'val_accuracies': [x['val_acc'] for x in history],
                            'epoch': np.arange(loaded_num_epochs)
                        }
                        df_acc = pd.DataFrame(data)

                        lines = alt.Chart(df_acc).mark_line().encode(
                            x='epoch',
                            y=alt.Y(alt.repeat("layer"), aggregate="mean", title="Accuracy"),
                            color=alt.datum(alt.repeat("layer"))
                        ).properties(title="Accuracy vs. Nº of epochs").repeat(
                            layer=["train_accuracies", "val_accuracies"])

                        lines.save(directoryOriginal + '/website/static/images/graphAccuracy.png')
                        #altair_chart(lines, use_container_width=True)


                    def plot_losses(history):
                        data = {
                            'train_losses': [x['train_loss'] for x in history],
                            'val_losses': [x['val_loss'] for x in history],
                            'epoch': np.arange(loaded_num_epochs)
                        }
                        df_acc = pd.DataFrame(data)

                        lines = alt.Chart(df_acc).mark_line().encode(
                            x='epoch',
                            y=alt.Y(alt.repeat("layer"), aggregate="mean", title="Losses"),
                            color=alt.datum(alt.repeat("layer"))
                        ).properties(title="Loss vs. Nº of epochs").repeat(layer=["train_losses", "val_losses"])

                        lines.save(directoryOriginal + '/website/static/images/graphLosses.png')
                        #altair_chart(lines, use_container_width=True)
                    

                    plot_accuracies(loaded_history)
                    plot_losses(loaded_history)
                    print(loaded_history)


    print("Session contents:", dict(session))
    return render_template("predictor.html", user=current_user, modelsSaved = session["modelsSaved"], existsModelsSaved=session["existsModelsSaved"],
                            buttonUseThisModel=session["buttonUseThisModel"], classPredicted=classPredicted,
                            catProbability=c, dogProbability=dogProbability, modelChosen=choose, test_accuracy_loaded=session['test_accuracy_loaded'] , tipo=session['tipo'], typeQuantum=session['typeQuantum'],
                            modelTL_loaded=session['modelTL_loaded'], optimizerTL_loaded=session['optimizerTL_loaded'], num_epochs_loaded=session['num_epochs_loaded'], lr_loaded=session['lr_loaded'],
                            pretrainedTL_loaded=session['pretrainedTL_loaded'], datasetTL_loaded=session['datasetTL_loaded'], validationSet_size_loaded=session['validationSet_size_loaded'],
                            batch_size_loaded=session['batch_size_loaded'], kernel_loaded=session['kernel_loaded'], gamma_loaded=session['gamma_loaded'], c_loaded=session['c_loaded'])


# ----------------------------------------------------------
# ----------------------- TRAINER -------------------------
# ----------------------------------------------------------

@views.route('/trainer', methods=['GET', 'POST'])
@login_required
def trainer():
    '''
    La función trainer() se encarga de recuperar todos las opciones elegidas por el usuario para el entrenamiento y de iniciar dicho entrenamiento en segundo plano mediante un nuevo hilo.
    '''
    global th
    global c
    global trainingFinished
    global stringPerformanceString
    global current_user_id
    global actualGraphImageName
    global previousGraphImageName
    global device

    
    c.terminate()

    previousGraphImageName = actualGraphImageName #para guardar el nombre de la imagen con el grafo anterior y poder eliminarlo a la hora de crear una imagen nueva
    actualGraphImageName = ""

    trainingFinished = False
    current_user_id = current_user.id

    stringPerformanceString = "" #para borrar la string de la performance del trainer cuando se sale del trainer

    session["buttonTrainModel"]= False

    if request.method == 'POST':

        c.setRunningTrue() #lo pongo a true el self._running para que la siguiente vez que de a "Train model" este a true y entre, si no entra directamente al BREAK

        session["buttonTrainModel"]= True

        datasetTL = request.form.get("selectBoxDataset")
        workWithTL = request.form.get("selectBoxWorkWith")
        modeTL = request.form.get("selectBoxMode")
        modelTL = request.form.get("selectBoxModelTL")
        pretrainedTL = request.form.get("selectBoxPretrained")
        svmTL = request.form.get("selectBoxSVM")
        optimizerTL = request.form.get("selectBoxOptimizer")
        validationSet_size = request.form.get("selectBoxValidationSize")
        lr = float(request.form.get("selectBoxLR"))
        num_epochs = int(request.form.get("rangeNumEpochs"))
        batch_size = int(request.form.get("batchSize"))
        
        cRegularizationSVM = float(request.form.get("cRegularizationSVM"))
        kernelSVM = request.form.get("selectKernelSVM")
        gammaSVM = request.form.get("selectGammaSVM")
        typeQuantum = request.form.get("selectBoxTypeQuantum")
        shuffleSVM = request.form.get("shuffleSVM")

        saveModel = None
        generatePDF = None
        if (request.form.get("saveModelSVM") is not None) or (request.form.get("saveModelQuantum") is not None) or (request.form.get("saveModelClassical") is not None):
            saveModel = 'on'
        if (request.form.get("generatePDFSVM") is not None) or (request.form.get("generatePDFQuantum") is not None) or (request.form.get("generatePDFClassical") is not None):
            generatePDF = 'on'


        session['datasetTL'] = datasetTL
        session['workWithTL'] = workWithTL
        session['modeTL'] = modeTL
        session['modelTL'] = modelTL
        session['pretrainedTL'] = pretrainedTL
        session['svmTL'] = svmTL
        session['optimizerTL'] = optimizerTL
        session['validationSet_size'] = validationSet_size
        session['lr'] = lr
        session['num_epochs'] = num_epochs
        session['batch_size'] = batch_size
        session['saveModel'] = saveModel
        session['generatePDF'] = generatePDF
        session['cRegularizationSVM'] = cRegularizationSVM
        session['kernelSVM'] = kernelSVM
        session['gammaSVM'] = gammaSVM
        session['typeQuantum'] = typeQuantum
        session['shuffleSVM'] = shuffleSVM

        #SI NO HAY GPU DISPONIBLE MOSTRAMOS MENSAJE INDICANDOLO
        device = str(get_default_device())
        if (device == 'cuda') & (workWithTL == 'GPU'):
            device = 'cuda'
        elif (device == 'cpu') & (workWithTL == 'GPU'):
            flash('Warning: GPU not available.', category='error')
            device = 'cpu'
            return render_template("trainer.html", user=current_user, buttonTrainModel=session["buttonTrainModel"])
        else:
            device = 'cpu'


        th = Thread(target= c.run, args=(datasetTL, workWithTL, modeTL, modelTL, pretrainedTL, svmTL,
                optimizerTL, validationSet_size, lr, num_epochs, batch_size, saveModel, generatePDF,cRegularizationSVM,kernelSVM,gammaSVM,shuffleSVM,typeQuantum,), daemon= True)

        
        th.start()
        
        #th.join() #para que el hilo principal espere a que haya terminado el hilo hijo y así no se recargue la pagina de trainer

        return redirect(url_for('views.trainingStarted'))


    return render_template("trainer.html", user=current_user, buttonTrainModel=session["buttonTrainModel"])




# ----------------------------------------------------------
# ----------------------- TRAINING STARTED -------------------------
# ----------------------------------------------------------

@views.route('/trainingStarted', methods=['GET', 'POST'])
@login_required
def trainingStarted():
    '''
    La función trainingStarted() se encarga de detectar cuando ha comenzado y terminado el entrenamiento y de ir actualziando la información mientras este dure.
    '''
    global c
    global randomNumTrainerGraphs
    global anteriorRandomNumTrainerGraphs
    
    #anteriorRandomNumTrainerGraphs = randomNumTrainerGraphs
    #randomNumTrainerGraphs = random.randint(1,1000000)
    #pathGraph = "{{url_for('static',filename='images/graphAccuracyTrainer" + str(randomNumTrainerGraphs) + ".png')}}"
    
    session['endTraining'] = False

    if th.is_alive():
        session['endTraining'] = False
    else:
        session['endTraining'] = True

    if request.method == 'POST':
        c.terminate()
        flash('Training cancelled!', category='success')
        return redirect(url_for('views.trainer'))

    return render_template("trainingStarted.html", user=current_user, datasetTL=session['datasetTL'], workWithTL=session['workWithTL'], modeTL=session['modeTL'],
                 modelTL=session['modelTL'], pretrainedTL=session['pretrainedTL'], svmTL=session['svmTL'],
                 optimizerTL=session['optimizerTL'], validationSet_size=session['validationSet_size'], lr=session['lr'], num_epochs=session['num_epochs'],
                 batch_size=session['batch_size'], saveModel=session['saveModel'], generatePDF=session['generatePDF'], cRegularizationSVM=session['cRegularizationSVM'], 
                 kernelSVM=session['kernelSVM'], gammaSVM=session['gammaSVM'], shuffleSVM=session['shuffleSVM'], typeQuantum=session['typeQuantum'], endTraining=session['endTraining'])
                 #randomNumTrainerGraphs=pathGraph)
    



# ----------------------------------------------------------
# ----------------------- HISTORY -------------------------
# ----------------------------------------------------------

@views.route('/history', methods=['GET', 'POST'])
@login_required
def history():
    '''
    La función history() se encarga de la gestión del historial del usuario, permitiendo alternar entre el listado de informes PDF guardados o el listado de modelos guardados. También gestiona la eliminación de alguno de dichos objetos.
    '''
    global c
    c.terminate()

    global listaModelsSaved

    stringPerformanceString = "" #para borrar la string de la performance del trainer cuando se sale del trainer

    session["existsPdfReports"] = True
    namesPdfs = {}
    urlsPdfs = {}
    session["namesPdfs"] = {}
    session["urlsPdfs"] = {}
    session["listaModelsSaved"] = {}
    

    if current_user.username == "admin":
        pathPDFreports = directoryOriginal + '/website/static/additional/pdfReports/' + str(current_user.username)
        pathModelsSaved = directoryOriginal + '/website/static/additional/modelsSaved/' + str(current_user.username)
    else:
        pathPDFreports = directoryOriginal + '/website/static/additional/pdfReports/user' + str(current_user.id)
        pathModelsSaved = directoryOriginal + '/website/static/additional/modelsSaved/user' + str(current_user.id)

    if request.method == 'POST':
        pdfEliminar = request.form['DeleteButton'] #'DeleteButton es el name de todos los botones Delete pero el request.form nos devuelve el value de ese boton que será el nombre del pdf
        #es el nombre del pdf para el que se ha pulsado el botón "Delete" y que debemos eliminar

        if pdfEliminar[:3] == 'pdf': #ELIMINAR PDF
            os.remove(pathPDFreports + '/' + pdfEliminar) #ELIMINO EL PDF
            user = User.query.filter_by(email=current_user.email).first()

            if user: 
                numTrainingsHistory = int(user.numTrainingsHistory)
                value = numTrainingsHistory - 1
                value = max(value,0)
                user.numTrainingsHistory = value #le resto uno al num de pdfs guardados por el usuario
                db.session.commit()
            
            reportPDF = ReportPDF.query.filter_by(name=pdfEliminar).first() #ELIMINO EL REPORT PDF DE LA BASE DE DATOS
            if reportPDF:
                db.session.delete(reportPDF) #se eliminba el usuario
                db.session.commit() #se comite el cambio

            flash('Training record successfully deleted!', category='success')

            

            
        
        else: #ELIMINAR MODEL SAVED
            os.remove(pathModelsSaved + '/' + pdfEliminar) #ELIMINO EL MODEL SAVED
            user = User.query.filter_by(email=current_user.email).first()

            if user: 
                numModelsSaved = int(user.numModelsSaved)
                value = numModelsSaved - 1
                value = max(value,0)
                user.numModelsSaved = value #le resto uno al num de pdfs guardados por el usuario
                db.session.commit()
            
            modelSaved = ModelSaved.query.filter_by(name=pdfEliminar).first() #ELIMINO EL REPORT PDF DE LA BASE DE DATOS
            if modelSaved:
                db.session.delete(modelSaved) #se eliminba el usuario
                db.session.commit() #se comite el cambio
            
            flash('Model successfully deleted!', category='success')
    

        #lo siguiente es para actualizar la lista automáticamente
        arr = os.listdir(pathPDFreports)

        counter = 0
        if len(arr)>0: 
            session["existsPdfReports"] = True
            while counter < len(arr):
                pdfName = str(arr[counter])
                if current_user.username == "admin":
                    urlThisPdf = '../static/additional/pdfReports/' + str(current_user.username) + '/' + pdfName #esta url se pasa al history.html para poder usarla en el <iframe> y mostrar el pdf
                else:
                    urlThisPdf = '../static/additional/pdfReports/user' + str(current_user.id) + '/' + pdfName
                
                temp = open(pathPDFreports + '/' + pdfName, 'rb')
                PDF_read = PdfReader(temp)
                num_pages = len(PDF_read.pages)
                temp.close()
                alturaCollapse = num_pages*1100
                session["namesPdfs"].update({pdfName : alturaCollapse})
                session["urlsPdfs"].update({pdfName : urlThisPdf})
                counter += 1
    
        else:
            session["existsPdfReports"] = False


            

        #para recuperar todos los modelos del usuario
        modelsSaved = ModelSaved.query.filter_by(user_id=current_user.id).all()

        
        if len(modelsSaved) == 0:
            session["existsModelsSaved"] = False
        else:
            session["existsModelsSaved"] = True
            listaModelsSaved = []
            for modelSaved in modelsSaved:
                listaModelsSaved.append(modelSaved) #guardo todos los modelos guardados de este usuario en una lista

        

    
    else:
        
        if not os.path.exists(pathPDFreports):
            session["existsPdfReports"] = False
            #st.warning("You do not have any previous training records. Select the Generate a PDF option report when training your models so that you can consult it here later.")
        else:
            arr = os.listdir(pathPDFreports)

            counter = 0
            if len(arr)>0:
                session["existsPdfReports"] = True
                while counter < len(arr):
                    pdfName = str(arr[counter])
                    if current_user.username == "admin":
                        urlThisPdf = '../static/additional/pdfReports/' + str(current_user.username) + '/' + pdfName #esta url se pasa al history.html para poder usarla en el <iframe> y mostrar el pdf
                    else:
                        urlThisPdf = '../static/additional/pdfReports/user' + str(current_user.id) + '/' + pdfName
                    
                    temp = open(pathPDFreports + '/' + pdfName, 'rb')
                    PDF_read = PdfReader(temp)
                    num_pages = len(PDF_read.pages)
                    temp.close()
                    alturaCollapse = num_pages*1100
                    session["namesPdfs"].update({pdfName : alturaCollapse})
                    session["urlsPdfs"].update({pdfName : urlThisPdf})
                    counter += 1
        
            else:
                session["existsPdfReports"] = False
        

            #para recuperar todos los modelos del usuario
            modelsSaved = ModelSaved.query.filter_by(user_id=current_user.id).all()

            
            if len(modelsSaved) == 0:
                session["existsModelsSaved"] = False
            else:
                session["existsModelsSaved"] = True
                listaModelsSaved = []
                for modelSaved in modelsSaved:
                    listaModelsSaved.append(modelSaved) #guardo todos los modelos guardados de este usuario en una lista


    return render_template("history.html", user=current_user, existsPdfReports=session["existsPdfReports"], namesPdfs= session["namesPdfs"], urlsPdfs=session["urlsPdfs"], existsModelsSaved=session["existsModelsSaved"], listaModelsSaved=listaModelsSaved)




# ----------------------------------------------------------
# ----------------------- PROFILE -------------------------
# ----------------------------------------------------------

@views.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    '''
    La función profile() recupera la información del usuario y se la muestra por medio de la vista Perfil.
    '''
    global c
    c.terminate()

    stringPerformanceString = "" #para borrar la string de la performance del trainer cuando se sale del trainer

    if request.method == 'POST':
            return redirect(url_for('views.editProfile'))
    

    return render_template("profile.html", user=current_user)



# ----------------------------------------------------------
# ----------------------- EDIT PROFILE -------------------------
# ----------------------------------------------------------

@views.route('/editProfile', methods=['GET', 'POST'])
@login_required
def editProfile():
    '''
    La función editProfile() muestra la vista Editprofile. En el caso de que el usuario realice cambios está hará efectivos dichos cambios.
    '''
    stringPerformanceString = "" #para borrar la string de la performance del trainer cuando se sale del trainer

    if request.method == 'POST':

        botonPulsado = request.form['editProfileButtons'] #obtiene si se ha pulsado el boton SaveChanges o el Delete Account
        if botonPulsado == "buttonDeleteAccount": #si se ha pulsado el botón DeleteAccount
            user = User.query.filter_by(username=current_user.username).first()
            if user:
                db.session.delete(user) #se eliminba el usuario
                db.session.commit() #se comite el cambio
            
            flash('Account deleted successfully!', category='success')
            return redirect(url_for('auth.login')) #redirige al usuario a la página admin

        else:
            print("entro aqui")
            fullname = request.form.get('fullname')
            email = request.form.get('email')
            username = current_user.username
            if current_user.username != "admin": #porque el admin no puede cambiar su username por lo que no se puede recoger el valor de una casilla de input que no existe
                username = request.form.get('username')

            password1 = request.form.get('password1')
            password2 = request.form.get('password2')
            

            user1 = User.query.filter_by(email=email).first() #obtiene ,si existe, el usuario con mismo email
            user2 = User.query.filter_by(username=username).first() #obtiene ,si existe, el usuario con mismo username
            if (user1 is not None) & (email != current_user.email): #que no existe nadie con ese mismo email excepto si eres tu mismo
                flash('Email already exists.', category='error')
            elif (user2 is not None) & (username != current_user.username): #que no existe nadie con ese mismo username excepto si eres tu mismo
                flash('User name already exists.', category='error')
            elif len(email) < 4:
                print("entro aqui3")
                flash('Email must be greater than 3 characters.', category='error')
            elif len(username) < 2:
                print("entro aqui4")
                flash('User name must be greater than 1 character.', category='error')
            elif password1 != password2:
                print("entro aqui5")
                flash('Passwords don\'t match.', category='error')
            elif len(password1) > 0 & len(password1) > 4:
                print("entro aqui7")
                flash('Password must be at least 4 characters.', category='error')
            else:
                if len(fullname) == 0:
                    fullname = "Not specified"
                
                if len(password1) == 0 | len(password2) == 0:
                    password1 = current_user.password
                
                print("entro aqui8")
                user = User.query.filter_by(email=current_user.email).first()

                if user:
                    print("entro aqui9")
                    user.fullname = fullname
                    user.email = email
                    user.username = username
                    user.password = generate_password_hash(password1)
                    db.session.commit() #comite los cambios

                    flash('Changes saved!', category='success')
                    return redirect(url_for('views.profile')) #redirige al usuario a la página profile
                else:
                    flash('An unexpected error has occurred, try again!', category='error')



    return render_template("editProfile.html", user=current_user)




# ----------------------------------------------------------
# ----------------------- ADMIN -------------------------
# ----------------------------------------------------------

@views.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    '''
    La función admin() muestra la vista Admin Settings. En el caso de que el administrador solicité crear un nuevo usuario, ver la información de un usuario, modificar un usuario o eliminar un usuario, está función hará efectivos los cambios.
    '''

    global c
    c.terminate()

    stringPerformanceString = "" #para borrar la string de la performance del trainer cuando se sale del trainer

    listaUsernames = []
    existsUsers = True
    flagBotonCreateNewUser = False
    flagBotonViewUserInfo = False
    flagBotonModifyUser = False
    flagBotonDeleteUser = False
    botonPulsado = None
    choose = None
    userChosen = None

    
    users = User.query.all()

    for user in users:
            listaUsernames.append(user.username)
    
    if len(listaUsernames) == 1:
        existsUsers = False


    if request.method == 'POST':
        botonPulsado = request.form['adminSettingsButton']
        #choose = request.form['RadioOptionsUsernames']
        choose = request.form.get('selectBoxAdminSettings')
        print(choose)
        userChosen = User.query.filter_by(username=choose).first()

        print(botonPulsado)
        if botonPulsado == 'createUserButton':
            flagBotonCreateNewUser = True
        elif botonPulsado == 'viewUserButton':
            flagBotonViewUserInfo = True
        elif botonPulsado == 'modifyUserButton':
            flagBotonModifyUser = True
        elif botonPulsado == 'deleteUserButton':
            flagBotonDeleteUser = True
        elif botonPulsado == 'createUserAdminButton': #boton para crear el usuario nuevo

            fullnameCreate = request.form.get('fullnameCreate')
            emailCreate = request.form.get('emailCreate')
            usernameCreate = request.form.get('usernameCreate')
            password1Create = request.form.get('password1Create')
            password2Create = request.form.get('password2Create')

            user1 = User.query.filter_by(email=emailCreate).first()
            user2 = User.query.filter_by(username=usernameCreate).first()
            if user1:
                flash('Email already exists.', category='error')
            elif user2:
                flash('User name already exists.', category='error')
            elif len(emailCreate) < 4:
                flash('Email must be greater than 3 characters.', category='error')
            elif len(usernameCreate) < 2:
                flash('User name must be greater than 1 character.', category='error')
            elif len(usernameCreate) > 20:
                flash('User name must be shorter than 21 characters.', category='error')
            elif password1Create != password2Create:
                flash('Passwords don\'t match.', category='error')
            elif len(password1Create) < 4:
                flash('Password must be at least 4 characters.', category='error')
            else:
                if len(fullnameCreate) == 0:
                    fullnameCreate = "Not specified"
                
                new_user = User(email=emailCreate, username=usernameCreate, password=generate_password_hash(password1Create), fullname= fullnameCreate, numModelsSaved= "0", numTrainingsHistory= "0")
                db.session.add(new_user) #crea un nuevo usuario
                db.session.commit() #comite los cambios

                #-------------------------------------
                #ENVÍO DE CORREO AL USUARIO CREADO
                #-------------------------------------
                message = "Hello, from the QKPDVSC team we inform you that the administrator has decided to create a new user account associated with this email.\n\n" + "Here are the details of your account to access the platform.\n\n" + "Since now," + "\n" + "your full name is: " + fullnameCreate + "\n" + "your registered email is: " + emailCreate + "\n" + "your user name is: " + usernameCreate + "\n" "your password is: " + password1Create + "\n\n" + "Best regards, the administrator."
                em = EmailMessage()
                em['From'] = 'alejandrossi2021@gmail.com'
                em['To'] = emailCreate
                em['Subject'] = "Changes in your QKPDVSC account"
                em.set_content(message)

                server = smtplib.SMTP('smtp.gmail.com',587)
                server.starttls()

                server.login('alejandrossi2021@gmail.com','igqshmdtmzveycdd')

                server.sendmail('alejandrossi2021@gmail.com', emailCreate, em.as_string())


                flash('User created successfully!', category='success')
                return redirect(url_for('views.admin')) #redirige al usuario a la página admin settings


        elif botonPulsado == 'modifyUserAdminButton': #boton para modificar el usuario 
            
            fullname = request.form.get('fullname')
            email = request.form.get('email')
            username = request.form.get('username')
            password1 = request.form.get('password1')
            password2 = request.form.get('password2')
            

            user1 = User.query.filter_by(email=email).first() #obtiene ,si existe, el usuario con mismo email
            user2 = User.query.filter_by(username=username).first() #obtiene ,si existe, el usuario con mismo username
            if (user1 is not None) & (email != userChosen.email): #que no existe nadie con ese mismo email excepto si eres tu mismo
                flash('Email already exists.', category='error')
            elif (user2 is not None) & (username != userChosen.username): #que no existe nadie con ese mismo username excepto si eres tu mismo
                flash('User name already exists.', category='error')
            elif len(email) < 4:
                flash('Email must be greater than 3 characters.', category='error')
            elif len(username) < 2:
                flash('User name must be greater than 1 character.', category='error')
            elif password1 != password2:
                flash('Passwords don\'t match.', category='error')
            elif len(password1) > 0 & len(password1) > 4:
                flash('Password must be at least 4 characters.', category='error')
            else:
                if len(fullname) == 0:
                    fullname = "Not specified"
                
                if len(password1) == 0 | len(password2) == 0:
                    password1 = userChosen.password
                
                user = User.query.filter_by(email=userChosen.email).first()

                if user:
                    oldEmail = userChosen.email
                    user.fullname = fullname
                    user.email = email
                    user.username = username
                    user.password = generate_password_hash(password1)
                    db.session.commit() #comite los cambios

                    #-------------------------------------
                    #ENVÍO DE CORREO AL USUARIO MODIFICADO
                    #-------------------------------------
                    if len(password1) == 0 | len(password2) == 0:
                        message = "Hello, there have been some changes to your QKPDVSC account.\n\n" + "Since now," + "\n" + "your full name is: " + fullname + "\n" + "your registered email is: " + email + "\n" + "your user name is: " + username + "\n" "your password is: The same as before"  + "\n\n" + "Best regards, the administrator."
                    else:
                        message = "Hello, there have been some changes to your QKPDVSC account.\n\n" + "Since now," + "\n" + "your full name is: " + fullname + "\n" + "your registered email is: " + email + "\n" + "your user name is: " + username + "\n" "your password is: " + password1 + "\n\n" + "Best regards, the administrator."
                    
                    em = EmailMessage()
                    em['From'] = 'alejandrossi2021@gmail.com'
                    em['To'] = oldEmail
                    em['Subject'] = "Changes in your QKPDVSC account"
                    em.set_content(message)

                    server = smtplib.SMTP('smtp.gmail.com',587)
                    server.starttls()

                    server.login('alejandrossi2021@gmail.com','igqshmdtmzveycdd')

                    server.sendmail('alejandrossi2021@gmail.com', oldEmail, em.as_string())



                    flash('Changes saved!', category='success')
                    return redirect(url_for('views.admin')) #redirige al usuario a la página admin




        elif botonPulsado == 'cancelDeleteAdminButton': #boton para cancelar la eliminacion de usuario

            return redirect(url_for('views.admin')) #redirige al usuario a la página admin settings

        elif botonPulsado == 'deleteUserAdminButton': #boton para eliminar el usuario
            
            user = User.query.filter_by(username=userChosen.username).first()
            if user:
                db.session.delete(user) #se eliminba el usuario
                db.session.commit() #se comite el cambio
            
            flash('User deleted successfully!', category='success')
            return redirect(url_for('views.admin')) #redirige al usuario a la página admin
        

    return render_template("admin.html", user=current_user, listaUsernames= listaUsernames, existsUsers = existsUsers, flagBotonCreateNewUser=flagBotonCreateNewUser,
    flagBotonViewUserInfo=flagBotonViewUserInfo, flagBotonModifyUser=flagBotonModifyUser, flagBotonDeleteUser=flagBotonDeleteUser, userChosen=userChosen)





@views.route('/update_flag', methods=['POST', 'GET'])
def update_flag():
    '''
    La función update_flag() se ejecuta constantemente cada medio segundo y su finalidad es proporcionar la información del entrenamiento en tiempo real a la plantilla trainingStarted.html para poder actualizar su información de manera dinámica sin tener que esperar al final del entrenamiento.
    '''

    global stringPerformanceString
    global trainingFinished
    global test_accuracy
    global test_loss
    global show_ModelSavedMessage
    global show_pdfReportMessage
    global actualGraphImageName
    global current_time_saveModel
    global current_time_pdfName
    global date
    global path
    global updateFlag
    global errorTraining
    restoDelContenido1 = ""
    restoDelContenido2 = ""
    flagShowGraphs = False
    

    if session['svmTL'] == 'YES': #SI ES MODELO TIPO SVM
        if trainingFinished == False:
            
            updateFlag = stringPerformanceString #FUNCIONA!!
            restoDelContenido1 = '''
            <p>&emsp;</p>
            <div id="loadingModelMessage" style="display: block;">
                <p><b>TRAINING</b></p>
                <div class="spinner-border">
                <span class="visually-hidden"></span>
                </div>
            </div>
            <p>&emsp;</p>
            <form method='POST'>
                <div class="text-lg-start mt-4 pt-2">
                    <button type="submit" class="btn btn-primary">Cancel Training</button> <!-- btn-primary son clases de BOOTSTRAP -->
                </div>
            </form>'''
            restoDelContenido2 = ''''''
            flagShowGraphs = False

        else:
            if errorTraining is None:
                updateFlag = stringPerformanceString
                restoDelContenido1 = ''' 
                <p>&emsp;</p>
                <div class="alert alert-success" role="alert" style="width: 200px;">
                    Training process finished succesfully!
                </div>
                '''
                
                restoDelContenido2 = '''
                <p>&emsp;</p>
                <p>&emsp;</p>
                <hr>
                <h4><b>MODEL PERFORMANCE</b></h4>
                <p>&emsp;</p>
                <p>When evaluating the model on the test dataset we obtain the following values.</p>
                <p>&ensp;</p>
                <p><b>Test accuracy:</b></p>
                <div class="alert alert-primary" role="alert" style="width: 200px;">
                    {:.3f}
                </div>
                <p>&ensp;</p>
                <p><b>Test loss:</b></p>
                <div class="alert alert-primary" role="alert" style="width: 200px;">
                    {:.3f}
                </div>'''.format(test_accuracy, test_loss)

                flagShowGraphs = False

            else:  #SI SE PRODUCE UN ERROR DURANTE EL ENTRENAMIENTO

                updateFlag = stringPerformanceString #FUNCIONA!!
                restoDelContenido1 = f'''
                <p>&emsp;</p>
                <div id="NoUsersWarning" class="alert alert-danger alter-dismissable fade show" role="alert">
                    <p>An exception occurred during training, so it will be necessary to make modifications and retrain the model.

                    <p><b>EXCEPTION OCURRED:</b></p>
                    <p>{errorTraining}</p>
                    <p>&ensp;</p>


                    <p><b>Solutions to the most common exceptions:</b></p>
                    <p>DefaultCPUAllocator: not enough memory --> The RAM capacity has been exceeded. It is necessary to reduce the Batch Size parameter.</p>
                    <p>OSError(28, 'No space left on device') --> The pre trained model could not be imported due to lack of space. Free up space on your device.</p> 
                </div>
                <p>&emsp;</p>
                '''
                restoDelContenido2 = ''''''
                flagShowGraphs = False


    elif session['modeTL'] == 'Quantum': #SI ES MODELO TIPO QUNATUM
        
        if session['typeQuantum'] == "VGG16 + Variational Quantum Circuit": #SI ES MODELO VARIATIONAL QUANTUM CIRCUIT
            if trainingFinished == False:
                
                updateFlag = stringPerformanceString #FUNCIONA!!
                restoDelContenido1 = '''
                <p>&emsp;</p>
                <p>&emsp;</p>
                <div id="loadingModelMessage" style="display: block;">
                    <p><b>TRAINING</b></p>
                    <div class="spinner-border">
                    <span class="visually-hidden"></span>
                    </div>
                </div>
                <form method='POST'>
                    <div class="text-lg-start mt-4 pt-2">
                        <button type="submit" class="btn btn-primary">Cancel Training</button> <!-- btn-primary son clases de BOOTSTRAP -->
                    </div>
                </form>'''
                restoDelContenido2 = ''''''
                flagShowGraphs = False

            else:
                if errorTraining is None:
                    updateFlag = stringPerformanceString
                    restoDelContenido1 = '''
                    <p>&emsp;</p>
                    <div class="alert alert-success" role="alert" style="width: 200px;">
                            Training process finished succesfully!
                    </div>
                    <p>&emsp;</p>
                    <p>&emsp;</p>
                    <hr>
                    <h4><b>RESULTS</b></h4>
                    <p>&emsp;</p>
                    '''
                    
                    restoDelContenido2 = '''
                    <p>&emsp;</p>
                    <p>&emsp;</p>
                    <hr>
                    <h4><b>MODEL PERFORMANCE</b></h4>
                    <p>&emsp;</p>
                    <p>When evaluating the model on the test dataset we obtain the following values.</p>
                    <p>&ensp;</p>
                    <p><b>Test accuracy:</b></p>
                    <div class="alert alert-primary" role="alert" style="width: 200px;">
                        {:.3f}
                    </div>
                    <p>&ensp;</p>
                    <p><b>Test loss:</b></p>
                    <div class="alert alert-primary" role="alert" style="width: 200px;">
                        {:.3f}
                    </div>'''.format(test_accuracy, test_loss)

                    flagShowGraphs = True

                else:  #SI SE PRODUCE UN ERROR DURANTE EL ENTRENAMIENTO

                    updateFlag = stringPerformanceString #FUNCIONA!!
                    restoDelContenido1 = f'''
                    <p>&emsp;</p>
                    <div id="NoUsersWarning" class="alert alert-danger alter-dismissable fade show" role="alert">
                        <p>An exception occurred during training, so it will be necessary to make modifications and retrain the model.

                        <p><b>EXCEPTION OCURRED:</b></p>
                        <p>{errorTraining}</p>
                        <p>&ensp;</p>


                        <p><b>Solutions to the most common exceptions:</b></p>
                        <p>DefaultCPUAllocator: not enough memory --> The RAM capacity has been exceeded. It is necessary to reduce the Batch Size parameter.</p>
                        <p>OSError(28, 'No space left on device') --> The pre trained model could not be imported due to lack of space. Free up space on your device.</p> 
                    </div>
                    <p>&emsp;</p>
                    '''
                    restoDelContenido2 = ''''''
                    flagShowGraphs = False
            
        else: #SI ES MODELO QUANTUM KERNEL

            if trainingFinished == False:
                
                updateFlag = stringPerformanceString #FUNCIONA!!
                restoDelContenido1 = '''
                <p>&emsp;</p>
                <div id="loadingModelMessage" style="display: block;">
                    <p><b>TRAINING</b></p>
                    <div class="spinner-border">
                    <span class="visually-hidden"></span>
                    </div>
                </div>
                <p>&emsp;</p>
                <form method='POST'>
                    <div class="text-lg-start mt-4 pt-2">
                        <button type="submit" class="btn btn-primary">Cancel Training</button> <!-- btn-primary son clases de BOOTSTRAP -->
                    </div>
                </form>'''
                restoDelContenido2 = ''''''
                flagShowGraphs = False


            else:
                if errorTraining is None:
                    updateFlag = stringPerformanceString
                    restoDelContenido1 = ''' 
                    <p>&emsp;</p>
                    <div class="alert alert-success" role="alert" style="width: 200px;">
                        Training process finished succesfully!
                    </div>
                    '''
                    
                    restoDelContenido2 = '''
                    <p>&emsp;</p>
                    <p>&emsp;</p>
                    <hr>
                    <h4><b>MODEL PERFORMANCE</b></h4>
                    <p>&emsp;</p>
                    <p>When evaluating the model on the test dataset we obtain the following values.</p>
                    <p>&ensp;</p>
                    <p><b>Test accuracy:</b></p>
                    <div class="alert alert-primary" role="alert" style="width: 200px;">
                        {:.3f}
                    </div>
                    <p>&ensp;</p>
                    <p><b>Test loss:</b></p>
                    <div class="alert alert-primary" role="alert" style="width: 200px;">
                        {:.3f}
                    </div>'''.format(test_accuracy, test_loss)

                    flagShowGraphs = False

                else:  #SI SE PRODUCE UN ERROR DURANTE EL ENTRENAMIENTO

                    updateFlag = stringPerformanceString #FUNCIONA!!
                    restoDelContenido1 = f'''
                    <p>&emsp;</p>
                    <div id="NoUsersWarning" class="alert alert-danger alter-dismissable fade show" role="alert">
                        <p>An exception occurred during training, so it will be necessary to make modifications and retrain the model.

                        <p><b>EXCEPTION OCURRED:</b></p>
                        <p>{errorTraining}</p>
                        <p>&ensp;</p>


                        <p><b>Solutions to the most common exceptions:</b></p>
                        <p>DefaultCPUAllocator: not enough memory --> The RAM capacity has been exceeded. It is necessary to reduce the Batch Size parameter.</p>
                        <p>OSError(28, 'No space left on device') --> The pre trained model could not be imported due to lack of space. Free up space on your device.</p> 
                    </div>
                    <p>&emsp;</p>
                    '''
                    restoDelContenido2 = ''''''
                    flagShowGraphs = False

    else: #SI ES MODELO TIPO NORMAL
    
        if trainingFinished == False:
            
            updateFlag = stringPerformanceString #FUNCIONA!!
            restoDelContenido1 = '''
            <p>&emsp;</p>
            <div id="loadingModelMessage" style="display: block;">
                <p><b>TRAINING</b></p>
                <div class="spinner-border">
                <span class="visually-hidden"></span>
                </div>
            </div>
            <form method='POST'>
                <div class="text-lg-start mt-4 pt-2">
                    <button type="submit" class="btn btn-primary">Cancel Training</button> <!-- btn-primary son clases de BOOTSTRAP -->
                </div>
            </form>'''
            restoDelContenido2 = ''''''
            flagShowGraphs = False

        else:
            if errorTraining is None:
                updateFlag = stringPerformanceString
                restoDelContenido1 = '''
                <p>&emsp;</p>
                <div class="alert alert-success" role="alert" style="width: 200px;">
                        Training process finished succesfully!
                </div>
                <p>&emsp;</p>
                <p>&emsp;</p>
                <hr>
                <h4><b>RESULTS</b></h4>
                <p>&emsp;</p>
                '''
                
                restoDelContenido2 = '''
                <p>&emsp;</p>
                <p>&emsp;</p>
                <hr>
                <h4><b>MODEL PERFORMANCE</b></h4>
                <p>&emsp;</p>
                <p>When evaluating the model on the test dataset we obtain the following values.</p>
                <p>&ensp;</p>
                <p><b>Test accuracy:</b></p>
                <div class="alert alert-primary" role="alert" style="width: 200px;">
                    {:.3f}
                </div>
                <p>&ensp;</p>
                <p><b>Test loss:</b></p>
                <div class="alert alert-primary" role="alert" style="width: 200px;">
                    {:.3f}
                </div>'''.format(test_accuracy, test_loss)

                flagShowGraphs = True

            else:  #SI SE PRODUCE UN ERROR DURANTE EL ENTRENAMIENTO

                updateFlag = stringPerformanceString #FUNCIONA!!
                restoDelContenido1 = f'''
                <p>&emsp;</p>
                <div id="NoUsersWarning" class="alert alert-danger alter-dismissable fade show" role="alert">
                    <p>An exception occurred during training, so it will be necessary to make modifications and retrain the model.

                    <p><b>EXCEPTION OCURRED:</b></p>
                    <p>{errorTraining}</p>
                    <p>&ensp;</p>

                    <p><b>Solutions to the most common exceptions:</b></p>
                    <p>DefaultCPUAllocator: not enough memory --> The RAM capacity has been exceeded. It is necessary to reduce the Batch Size parameter.</p>
                    <p>OSError(28, 'No space left on device') --> The pre trained model could not be imported due to lack of space. Free up space on your device.</p> 
                </div>
                <p>&emsp;</p>
                '''
                restoDelContenido2 = ''''''
                flagShowGraphs = False


    user = User.query.filter_by(email=current_user.email).first()

    if user:
        if show_ModelSavedMessage == True:
            numModelsSaved = int(user.numModelsSaved)
            user.numModelsSaved = numModelsSaved + 1 #aumento el numero de modelos guardados del usuario

            if show_pdfReportMessage == True: #si tiene un pdf que se crea a la vez, un pdf asociado
                new_modelSaved = ModelSaved(name=current_time_saveModel, date=date, test_accuracy=round(test_accuracy,2), test_loss=round(test_loss,2), reportPDFassociated = current_time_pdfName, path=path, user_id=current_user.id)
            else:
                new_modelSaved = ModelSaved(name=current_time_saveModel, date=date, test_accuracy=round(test_accuracy,2), test_loss=round(test_loss,2), reportPDFassociated = 'No PDF report associated', path=path, user_id=current_user.id)
            db.session.add(new_modelSaved) #crea un nuevo ModelSaved
            db.session.commit() #comite los cambios

            flash("The model have been saved in your user folder inside the modelsSaved folder!", category='success')
            show_ModelSavedMessage = False

        if show_pdfReportMessage == True:
            numTrainingsHistory = int(user.numTrainingsHistory)
            user.numTrainingsHistory = numTrainingsHistory + 1 #aumento el numero de informes pdf del usuario

            new_reportPDF = ReportPDF(name=current_time_pdfName, date=date, path=path, user_id=current_user.id)
            db.session.add(new_reportPDF) #crea un nuevo reportPDF
            db.session.commit() #comite los cambios

            flash("The PDF report have been generated in your folder inside the pdfReports folder! You can also consult it in the History section", category='success')
            show_pdfReportMessage = False

        #db.session.commit()#como aquí ya habría terminado el entrenamiento, si se seleccionó Save Model o
        # generate PDF, aumento en 1 el numero de modelsSaved o de trainings in history
        
    
    dictionary_to_return = {
        'updateFlag': updateFlag,
        'restoDelContenido1': restoDelContenido1,
        'restoDelContenido2': restoDelContenido2,
        'flagShowGraphs': flagShowGraphs,
        'actualGraphImageName': actualGraphImageName
    }
    return jsonify(dictionary_to_return)


@views.route('/upload_dropzone', methods=['GET', 'POST'])
def upload_dropzone():
    '''
    La función upload_dropzone() se encraga de comprobar si se ha subido un archivo válido al elmento Drag and drop de la vista Predictor. En caso de ser un archivo válido, lo almacena en el sistema.
    '''
    global imageUploaded
    global counterImages
    imageUploaded = False

    if request.method == 'POST':
        f = request.files.get('file')
        
        
        numeroPuntos = len(f.filename.split('.'))

        if (f.filename.split('.')[numeroPuntos-1] != 'png') & (f.filename.split('.')[numeroPuntos-1] != 'jpg') & (f.filename.split('.')[numeroPuntos-1] != 'jpeg'):
            return 'PNG, JPG or JPEG only!', 400  # return the error message, with a proper 4XX code
        
        f.save(os.path.join(directoryOriginal + '/website/static/images', "predictionImage.jpg"))
        imageUploaded = True
        counterImages += 1

    return 'upload template'


