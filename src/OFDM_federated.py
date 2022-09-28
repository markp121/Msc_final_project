'''OFDM_split.py uses a modified version of the federated learning implementation from https://github.com/MahdiBoloursazMashhadi/FedRec for the paper:
"FedRec: Federated Learning of Universal Receivers over Fading Channels" - M., B. Mashhadi et al. 
This code roughly follows the TensorFlow Federated library tutorial for machine learning with federated datasets found here: 
https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification
Dataset generation code adapted from the scripts in GitHub repsository: https://github.com/haoyye/OFDM_DNN for the paper:
"Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems" - H. Ye , G. Ye Li, and B. Juang
'''

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_federated as tff

from collections import OrderedDict
from OFDM_funtions import *

import os
from sys import argv
from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    """Parses CL arguments

    Returns:
        Namespace object containing all arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-bs", "--batch_size", type=int, default=100)
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=10000)
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=10000)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-spd", "--samples_per_device", type=int, default=20000)
    parser.add_argument("-ts", "--test_size", type=int, default=100000)
    parser.add_argument('-vs', "--val_size", type=int, default=10000)
    parser.add_argument("-snr", "--SNR", type=float, default=25)
    parser.add_argument("-p", "--pilots", type=int, default=64)
    parser.add_argument("-u", "--users", type=int, default=5)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-r", "--rounds", type=int, default=10)
    parser.add_argument("-t", "--trials", type=int, default=10)
    return parser.parse_args(argv[1:])


def data_gen(channel_data, SNRdb, dset):
    if dset == 'train':
        size = args.samples_per_device
    elif dset == 'val':
        size = args.val_size
    elif dset == 'test':
        size = args.test_size
    
    index = np.random.choice(np.arange(len(channel_data)), size)  # Select a random sample without replacement of the channel vectors to form the dataset
    H_total = channel_data[index]  # Extract those channel vectors from the full dataset
    input_samples = np.zeros((size,256), dtype=np.float32)
    input_labels = np.zeros((size,16), dtype=np.float32)
    for i, H in enumerate(H_total):
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))  # generate random data carrying bits
        signal_output, sent_symbols = ofdm_simulate(bits, H, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers)  # generate model inputs and labels
        input_labels[i, :] = sent_symbols
        input_samples[i, :] = signal_output
    return np.asarray(input_samples), np.asarray(input_labels)


def create_dataset(path, train_or_test, SNRdb):

    data = np.load(path)  # load data from file

    ## Create data loaders for tensorflow 
    if train_or_test == 'train':
        inputs, labels = data_gen(data, SNRdb, dset=train_or_test)
        return tf.data.Dataset.from_tensor_slices((inputs, labels))
    elif train_or_test == 'val':
        inputs, labels = data_gen(data, SNRdb, dset=train_or_test)
        return tf.data.Dataset.from_tensor_slices((inputs, labels))
    elif train_or_test == 'test':
        inputs, labels = data_gen(data, SNRdb, dset=train_or_test)
        return tf.data.Dataset.from_tensor_slices((inputs, labels))

## Custom loss metric for for use with TFF
class BitErrorRate(tf.keras.metrics.Metric):
    
    def __init__(self, name="bit_error_rate", **kwargs):
        super(BitErrorRate,self).__init__(name=name, **kwargs)
        self.bit_errors = self.add_weight('bit_errors', initializer='zeros')
        self.batch = self.add_weight('batch', initializer='zeros')

    ## Reset variables after every round of evaluation
    def reset_state(self):
        self.bit_errors.assign(0)
        self.batch.assign(0)

    ## Calculate bit error rate from the predicted labels and true labels 
    def bit_error(self, y_true, y_pred):
        y_pred = tf.sign(y_pred)
        return 1 - tf.reduce_mean(tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32),1))

    ## Update state after each batch
    def update_state(self, y_true, y_pred):
        self.bit_errors.assign_add(self.bit_error(y_true, y_pred))
        self.batch.assign_add(1)

    ## Return metric for all data points in the set
    def result(self):
        return self.bit_errors / self.batch

## Shuffle, format and make the dataset compatible with TFF
def preprocess(dataset, batch_size):
    def batch_format_fn(element1,element2):
        return OrderedDict(
            x=tf.reshape(element1, [-1, 256]),
            y=tf.reshape(element2, [-1, 16]))
    ## Create TF dataset split into suffled batches, repeated for the local epochs
    return dataset.repeat(local_epochs).shuffle(shuffle_buffer).batch(
      batch_size).map(batch_format_fn).prefetch(prefetch_buffer)

## Model architecture
def create_keras_model():
    return tf.keras.models.Sequential([
        layers.InputLayer(input_shape=(256,)),
        layers.Dense(500, activation='relu'),
        layers.Dense(250, activation='relu'),
        layers.Dense(120, activation='relu'),
        layers.Dense(16),])

## Define model function
def model_fn():
  keras_model = create_keras_model()  # Initilize sequential model
  return tff.learning.from_keras_model(
    keras_model,
    input_spec=val_data[0].element_spec,
    loss=tf.keras.losses.MeanSquaredError(),  # MSE error for training
    metrics=[BitErrorRate()])  # custom bit error metric class
    

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    args = parse_args()

    # Training Parameters
    users = args.users  # number of users taking part in federated training
    local_epochs = args.epochs  # number of local epochs for each aggragation round
    aggregation_rounds = args.rounds  # number of federated aggregation rounds
    shuffle_buffer = 1000 
    prefetch_buffer = 10
    train_SNRdb = args.SNR  # SNR per bit value in decibels

    trials = args.trials

    K = 64  # Number of sub-carriers per OFDM symbol
    CP = K//4  # Length of cyclic prefix
    P = args.pilots # Number of pilot symbols
    allCarriers = np.arange(K)  # Indices of subcarriers
    mu = 2  # Number of OFDM symbols in a frame
    CP_flag = True  # Choose to include a cyclic prefix or not

    if P < K:
        pilotCarriers = allCarriers[::K//P]  # Pilots are every (K/P)th carrier if there are less that K pilots
        dataCarriers = np.delete(allCarriers, pilotCarriers)  # Delete pilots from the data carrier vector
    else:
        pilotCarriers = allCarriers  # All sub-carriers hold a piot symbol in the first OFDM symbol
        dataCarriers = [] 

    payloadBits_per_OFDM = K * mu  # Total number of sub-carrier symbols in the frame
    
    ## Load pilots from txt file or create a new one with random bits that are kept constant in training and testing
    Pilot_file_name = 'Pilot_'+str(P)
    if os.path.isfile(Pilot_file_name):
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        np.savetxt(Pilot_file_name, bits, delimiter=',')

    pilotValue = Modulation(bits, mu)  # Create QPSK symbols from the pilot bits

    SNRs = [5,10,15,20,25]
    results = {snr: 0.0 for snr in SNRs}

    for i in range(trials):
        val_data = [preprocess(create_dataset('../data/channel_test_full.npy', "val", train_SNRdb), args.val_batch_size)]

        ## Create the federated datasets for each user
        federated_data = []
        for u in range(users):
            user_data = create_dataset('../data/channel_test_full.npy', "train", train_SNRdb)
            federated_data.append(preprocess(user_data, args.batch_size))

        ## Initialize federated averaging algorithm for training
        iterative_process = tff.learning.build_federated_averaging_process(model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
                                                                                     server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

        ## Initialize evaluation object
        evaluation = tff.learning.build_federated_evaluation(model_fn)

        state = iterative_process.initialize()  # Initialize first federated state
        for n in range(aggregation_rounds):
            state, metrics = iterative_process.next(state, federated_data)  # complete local epoch and update global model (state) 
            print("aggregation round: {} - {}".format(n+1, metrics['train']))
            val_metrics = evaluation(state.model, val_data)  # Validation
            print("Validation results: ", val_metrics)

        ## test across SNR range
        for snr in SNRs:
            test_data = [preprocess(create_dataset('../data/channel_test_full.npy', "test", snr), args.test_batch_size)]
            test_metrics = evaluation(state.model, test_data)
            results[snr] += test_metrics["bit_error_rate"] / trials
        
    print("Results with {} pilots:\n {}".format(P, results))

