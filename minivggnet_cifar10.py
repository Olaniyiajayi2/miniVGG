import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.simpledatasetloader import SimpleDatasetLoader
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt 
import argparse
import numpy as np 
from minivggnet import MiniVggNet


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "path to save plots")
args = vars(ap.parse_args())


print("[INFO] loading cifar_10...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY - lb.fit_transform(testY)


# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVggNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
        batch_size=64, epochs=40, verbose=1)


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 64)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = [str(x) for x in lb.classes_]))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(o, 40), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])