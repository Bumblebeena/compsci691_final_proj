import csv
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    in_csv = "train_log_2022-04-30_19-02.csv"
    with open(in_csv) as in_c:
        csvreader = csv.reader(in_c)
        next(csvreader)
        epochs = []
        accuracies = []
        losses = []
        mean_ious = []
        for epoch, accuracy, loss, mean_iou in csvreader:
            
            epochs.append(epoch)
            accuracies.append(accuracy)
            losses.append(loss)
            mean_ious.append(mean_iou)

    epochs = np.asarray(epochs, dtype='float32')
    accuracies = np.around(np.asarray(accuracies, dtype='float32'), 3)
    losses = np.around(np.asarray(losses, dtype='float32'), 3)
    mean_ious = np.around(np.asarray(mean_ious, dtype='float32'), 3)

    plt.plot(epochs, accuracies)
    plt.title("PARCnet Pixel Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig("output/accuracy.png")
    plt.close()

    plt.plot(epochs, losses)
    plt.title("PARCnet Pixel Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("output/loss.png")
    plt.close()

    plt.plot(epochs, accuracies)
    plt.title("PARCnet Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.grid()
    plt.savefig("output/mean_iou.png")
    plt.close()
