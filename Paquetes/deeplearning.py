from scikeras.wrappers import KerasClassifier 
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import F1Score, Recall,Precision 
from keras.utils import to_categorical # Necesario para que cada etiqeuta de sample sea codificada como (3,1) siendo este un 
                                       # un vector binario con un 1 asociado a la etiqueta de la clase 
from sklearn.preprocessing import label_binarize
def baseline_model():
    # Create model here
    model = Sequential()
    model.add(Dense(14, input_dim = 18, activation = 'relu')) # Rectified Linear Unit Activation Function
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(25, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(
                  loss = 'categorical_crossentropy', 
                  optimizer = 'adam', 
                  metrics = [
                              F1Score(average="weighted", threshold=None, name="f1_score", dtype=None),
                              Recall( class_id = 0, name="Recall_class_0"),
                              Recall( class_id = 1, name="Recall_class_1"),
                              Recall( class_id = 2, name="Recall_class_2"),
                              Precision( class_id = 0, name="Precision_class_0"),
                              Precision( class_id = 1, name="Precision_class_1"),
                              Precision ( class_id = 2, name="Precision_class_2"),
                            
                            ]
                  )
    return model

NN_clf = KerasClassifier(build_fn = baseline_model, epochs = 100, batch_size = 10, verbose = 1)