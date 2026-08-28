import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from PIL import Image
from sklearn.preprocessing import LabelEncoder 
import seaborn as sns 
import cv2






# function to run the model and show results 
def run():

    file_path = os.path.dirname(__file__)
    dataset_path = os.path.join(file_path, 'Data')

    
    image_paths = []
    labels = []
    data = []
    
    
    
    encoder = LabelEncoder()
    
    # preprocessing 
    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)
        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                image_paths.append(img_path)
                labels.append(label)
    
                pic = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                clahe = cv2.createCLAHE(clipLimit=5)
                pic = clahe.apply(pic)
    
                pic = Image.fromarray(pic)
    
                pic = pic.resize((224, 224))
                pic = pic.convert("L")
    
                img_array = np.array(pic)
                img_flat = img_array.flatten()
                data.append(img_flat)      
        
    
    
    encoded_labels = encoder.fit_transform(labels)
    
    X = np.array(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        encoded_labels,
        test_size = 0.3,
        stratify=labels,
        random_state = 42
        )
    
    
    
    percep = Perceptron(max_iter = 1000, eta0=0.1)
    #train the model
    percep.fit(X_train, y_train)
    #test the model
    y_pred = percep.predict(X_test)
    
    #metrics calculation
    accuracymultipercep = accuracy_score(y_test, y_pred,)
    print('Accuracy =', accuracymultipercep)
    print('')
    precisionmultipercep = precision_score(y_test, y_pred, average='weighted')
    print('Precision =', precisionmultipercep)
    print('')
    recallmultipercep = recall_score(y_test, y_pred, average='weighted')
    print('Recall =', recallmultipercep)
    print('')
    f1multipercep = f1_score(y_test, y_pred, average='weighted')
    print('F1 score =', f1multipercep)
    
    conMatrix = confusion_matrix(y_test, y_pred)
    acc_per_class = conMatrix.diagonal() / conMatrix.sum(axis=1)
    
    # plot metrics 
    plt.figure(figsize=(6,5))
    sns.heatmap(conMatrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0 Normal', '1 Doubtful', '2 Mild', '3 Moderate', '4 Severe'],
                yticklabels=['0 Normal', '1 Doubtful', '2 Mild', '3 Moderate', '4 Severe'])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (Simple Perceptron)")
    plt.show()
    plt.close()
    
    i = 0;
    for x in acc_per_class:
        print("Accuracy of class ", i, " ", acc_per_class[i])
        i = i + 1