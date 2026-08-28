import os
import tensorflow
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import cv2

import seaborn as sns
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping



# function to train the CNN and output the results 
def run():

    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
        )
    
    
    #freeze convolutional base
    conv_base.trainable = False
    
    
    file_path = os.path.dirname(__file__)
    dataset_path = os.path.join(file_path, 'Data')

    
    image_paths = []
    labels = []
    data = []
    
    encoder = LabelEncoder()
    
    # image preprocessing 
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
                img_array = np.expand_dims(img_array, axis=-1)
                img_rgb = tensorflow.image.grayscale_to_rgb(tensorflow.convert_to_tensor(img_array))
                #img_flat = img_array.flatten()
                data.append(img_rgb)
    
    
    # encoding written labels 
    encoded_labels = encoder.fit_transform(labels)
    
    
    data = tensorflow.stack(data)
    labels = np.array(labels)
    
    
    X = np.array(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        encoded_labels,
        test_size = 0.3,
        stratify=labels,
        random_state = 42
        )
    
    #build model
    model = models.Sequential([
        conv_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), #helps prevent overfitting
        layers.Dense(5, activation='softmax')]) # categorical classification output
    
    
    
    
    # network and training parameters
    NB_EPOCH = 25
    BATCH_SIZE = 16
    OPTIMIZER = Adam(learning_rate=0.001)
    VALIDATION_SPLIT = 0.2
    IMG_ROWS, IMG_COLS = 224, 224  # input image dimensions
    NB_CLASSES = 5  # number of outputs = number of digits
    INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, NB_CLASSES)
    y_test = to_categorical(y_test, NB_CLASSES)
    
    
    # initialise the optimiser and model
    
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
        )
    
    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=OPTIMIZER,
                  metrics=["accuracy",
                  tensorflow.keras.metrics.Precision(name='precision'),
                  tensorflow.keras.metrics.Recall(name='recall')]
                  )
    # train the model
    history_pretrain = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=NB_EPOCH,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stop]
    )
    
    # Make the base convolutional model trainable from a certain layer onwards
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
      if layer.name == 'block5_conv1':  # start fine-tuning from this layer
        set_trainable = True
      layer.trainable = set_trainable  # layers before this remain frozen, layers from here on are trainable
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=["accuracy",
                  tensorflow.keras.metrics.Precision(name='precision'),
                  tensorflow.keras.metrics.Recall(name='recall')]
                  )
    
    history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=10,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stop]
    )
    
    # evaluate on test data
    score = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    cm = confusion_matrix(y_true, y_pred)
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    
    # plot metrics 
    plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0 Normal', '1 Doubtful', '2 Mild', '3 Moderate', '4 Severe'],
                yticklabels=['0 Normal', '1 Doubtful', '2 Mild', '3 Moderate', '4 Severe'])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (Transfer Learning)")
    
    
    best_val_acc = max(history.history['val_accuracy'])
    best_val_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    best_train_acc = max(history.history['accuracy'])
    best_train_epoch = history.history['accuracy'].index(best_train_acc) + 1
     
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("Best validation accuracy: ", best_val_acc, " at epoch ", best_val_epoch)
    print("Best training accuracy: ", best_train_acc, " at epoch ", best_train_epoch)
    print("F1 score: ", f1)
    print("Precision: ", score[2])
    print("Recall: ", score[3])
    # list all metrics recorded during training
    
    print(history.history.keys())
    print('')
    
    i = 0;
    for x in acc_per_class:
        print("Accuracy of class ", i, " ", acc_per_class[i])
        i = i + 1
    
    plt.subplot(2,2,2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy (Transfer Learning)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
   
    
    plt.subplot(2,2,3)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (Transfer Learning)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
  
    
    plt.subplot(2,2,4)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Model Precision and Recall (Transfer Learning)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    


    plt.show()
    
  













