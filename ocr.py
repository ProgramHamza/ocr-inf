from tqdm import tqdm

a = input("Input imag name:")




    # This code only exists to help us visually inspect the images.
    # It's in an `if DEBUG:` block to illustrate that we don't need it for our code to work.
from PIL import Image
import numpy as np

def read_image(path):
     return np.asarray(Image.open(path).convert('L'))



DATA_DIR = "C:/Users/mhrom/Downloads/data"
TEST_DATA_FILENAME = DATA_DIR + "/t10k-images.idx3-ubyte"
TEST_LABELS_FILENAME = DATA_DIR + "/t10k-labels.idx1-ubyte"
TRAIN_DATA_FILENAME = DATA_DIR + "/train-images.idx3-ubyte"
TRAIN_LABELS_FILENAME = DATA_DIR + "/train-labels.idx1-ubyte"



def bytes_to_int(b):
    return int.from_bytes(b,'big')

def read_images(filename,n_max_images):
    images = []
    with open(filename, "rb") as f:
        _ = f.read(4) # magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_colums = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_colums):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

def flatten_list():
    pass

def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels

def max_frequency(l):
    return max(l, key=l.count)

def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

def extract_features(X):
    return [flatten_list(sample) for sample in X]

def dist(a,b):
    return sum(
        [
        (bytes_to_int(a_i) - bytes_to_int(b_i))**2 for a_i, b_i in zip(a,b)
        ]
    )**(0.5)

def get_training_distance_for_sample(test_sample, X_train):
    return [dist(train_sample, test_sample) for train_sample in X_train]

def knn(X_train, Y_train, X_test, Y_test, k=3):
    y_pred = []
    for sample_idx, sample in tqdm(enumerate(X_test), desc="Running k-NN", total=len(X_test), unit="sample"):
        distances = get_training_distance_for_sample(sample, X_train)
        sorted_distances_indices = [
            pair[0]
            for pair in sorted(enumerate(distances), key=lambda x: x[1])]
        k_nearest_indicies = sorted_distances_indices[:k]
        k_nearest_labels = [Y_train[i] for i in k_nearest_indicies]
        top_guess = max_frequency(k_nearest_labels)
        y_pred.append(top_guess)
    return y_pred

def main():
    X_train = read_images(TRAIN_DATA_FILENAME,2000)
    Y_train = read_labels(TRAIN_LABELS_FILENAME)
    Y_test = read_labels(TEST_LABELS_FILENAME)

    X_test = [read_image(f"C:/Users/mhrom/Downloads/{a}.png")]

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)
    
    y_pred = knn(X_train, Y_train, X_test, Y_test,k=3)
    

    accuracy = sum([
        int(y_pred_i == y_test_i)
        for y_pred_i, y_test_i
        in zip(y_pred, Y_test)
    ]) / len(Y_test)


    print(f'Accuracy: {accuracy * 100}%')



if __name__ == "__main__":
    main()
