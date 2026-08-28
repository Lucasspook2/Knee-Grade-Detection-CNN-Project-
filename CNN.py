import os
import tensorflow
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, f1_score
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder 
import seaborn as sns 
from tensorflow.keras.callbacks import EarlyStopping
import cv2


pool_layer = MaxPooling2D(pool_size=(2, 2), 
strides=(2, 2))



conv_layer = Conv2D(filters=32, kernel_size=(3, 3), 
padding='valid')

class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        Input(shape=input_shape)
        
        # CONV => RELU => POOL
        model.add(Conv2D(20, kernel_size=5, padding="valid"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # CONV => RELU => POOL
        model.add(Conv2D(50, kernel_size=5, padding="valid"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Flatten => RELU layers
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Activation("relu"))
        #add dropout to prevent overfitting 
        model.add(Dropout(0.5))
       # a softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        
        return model



#function to train the CNN and ouput the test results 
def run():
    file_path = os.path.dirname(__file__)
    dataset_path = os.path.join(file_path, 'Data')

    
    image_paths = []
    labels = []
    data = []
    
    
    
    encoder = LabelEncoder()
    
    """
    
    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)
        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                image_paths.append(img_path)
                labels.append(label)
                
                pic = Image.open(img_path)
                pic = pic.resize((256, 256))
                pic = pic.convert("L")
                
                img_array = np.array(pic)
                img_flat = img_array.flatten()
                data.append(img_flat)
     """

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
                img_flat = img_array.flatten()
                data.append(img_flat)      
        
    
    #encoding the written lables for use in the CNN
    encoded_labels = encoder.fit_transform(labels)
    
    X = np.array(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        encoded_labels,
        test_size = 0.3,
        stratify=labels,
        random_state = 42
        )
        
    # network and training parameters
    NB_EPOCH = 50
    BATCH_SIZE = 16
    OPTIMIZER = Adam(learning_rate=0.001)
    VALIDATION_SPLIT = 0.2
    IMG_ROWS, IMG_COLS = 224, 224  # input image dimensions
    NB_CLASSES = 5  # number of outputs = number of digits
    INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
    
    #normalising data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # reshape to [samples, rows, cols, channels] for Conv2D
    X_train = X_train.reshape(-1, IMG_ROWS, IMG_COLS, 1)
    X_test = X_test.reshape(-1, IMG_ROWS, IMG_COLS, 1)
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, NB_CLASSES)
    y_test = to_categorical(y_test, NB_CLASSES)
    
    
    # initialise the optimiser and model
    model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    # compile the model
    model.compile(
    loss="categorical_crossentropy",
    optimizer=OPTIMIZER,
    metrics=["accuracy",
    tensorflow.keras.metrics.Precision(name='precision'),
    tensorflow.keras.metrics.Recall(name='recall')]
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
        )
    
    
    
    
    # train the model
    history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=NB_EPOCH,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stop]
    )
    # evaluate on test data
    score = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print('cm ', cm)
    
    # per class accuracy 
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # best validation accuracy and training accuracy 
    best_val_acc = max(history.history['val_accuracy'])
    best_val_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    best_train_acc = max(history.history['accuracy'])
    best_train_epoch = history.history['accuracy'].index(best_train_acc) + 1
    
    
    # plotting of metrics 
    plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0 Normal', '1 Doubtful', '2 Mild', '3 Moderate', '4 Severe'],
                yticklabels=['0 Normal', '1 Doubtful', '2 Mild', '3 Moderate', '4 Severe'])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (CNN)")
    
    
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("Best validation accuracy: ", best_val_acc, " at epoch ", best_val_epoch)
    print("Best training accuracy: ", best_train_acc, " at epoch ", best_train_epoch)
    # list all metrics recorded during training
    print("F1 score: ", f1)
    print("Precision: ", score[2])
    print("Recall: ", score[3])
    print('')
    
    i = 0;
    for x in acc_per_class:
        print("Accuracy of class ", i, " ", acc_per_class[i])
        i = i + 1
    
    print(history.history.keys())
    
    plt.subplot(2,2,2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy (CNN)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
   
    
    plt.subplot(2,2,3)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (CNN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
  
    
    plt.subplot(2,2,4)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Model Precision and Recall (CNN)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    


    plt.show()
