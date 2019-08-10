from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np
import itertools

def get_test_predictions(model, x_test):
    """
    Gets predictions given test data (array) and a model. Returns array of predictions.
    """
    predictions = model.predict(x_test)
    y_pred = []
    for pred in predictions:
        pred = list(pred)
        y_pred.append(pred.index(max(pred)))
    return y_pred

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
	- Code taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html -
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Wahre Klasse')
    plt.xlabel('Vorhergesagte Klasse')
    plt.tight_layout()

def plot_accuracy(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['acc']
    epochs = range(1, len(acc)+1)

    plt.clf()
    val_acc = history_dict['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
	
def print_f1_scores(y_test, y_pred):
    macro = f1_score(y_test, y_pred, average='macro')  

    micro = f1_score(y_test, y_pred, average='micro')  

    weighted = f1_score(y_test, y_pred, average='weighted')
    print('F1 Score')
    print("macro:\t",macro, '\tmicro:\t', micro, '\tweighted:\t', weighted )
    return weighted
	
def save_f1_scores(f1_array, output_file):
    with open(output_file, 'w') as f:
        for score in f1_array:
            f.write("%s\n" % score)
        f.write("\n")
        f.write("Average: %s"  % np.mean(f1_array))