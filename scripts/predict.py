from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from process_data_3d_generator import ProcessData
from keras.models import load_model
import numpy as np
import pickle

def scores(model, X_test, y_test):
    y_pred = model.predict_classes(X_test)
    y_true = []
    for row in y_test:
        y_true.append(row.argmax())
    y_true = np.array(y_true)
    print(classification_report(y_true, y_pred))
    print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
    return (y_pred, classification_report(y_true, y_pred), accuracy_score(y_true, y_pred), recall_score(y_true, y_pred, average=None), precision_score(y_true, y_pred, average=None), f1_score(y_true, y_pred, average=None))

def predict_class(model, X):
    '''Pass in an individual image to predict classes'''

    predict_probability = model.predict_proba(X)
    predict_label = model.predict_classes(X)
    return predict_probability, predict_label



if __name__ == '__main__':
    # test_data = ProcessData(25, (200,200))
    # X_test, y_test = test_data.generate_images_in_memory('test', avg=False, BW=False)
    # img = np.expand_dims(X_test[6], axis=0)
    # model_lrcn = load_model('../lrcn_model.h5')
    # y_pred, classification, accuracy, recall, precision, f1 = scores(model_lrcn, X_test, y_test)
    # predict_proba, prediction = predict_class(model_lrcn, img)
