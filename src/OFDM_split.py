'''OFDM_split.py uses a modified version of the split learning implementation from:
"Scalable Machine Learning ID2223 Project: An Investigation Into Split Learning" - X. Ioannidou, B. T. Straathof.
Dataset generation code adpated from the git hub repsository: https://github.com/haoyye/OFDM_DNN for the paper:
"Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems" - H. Ye , G. Ye Li, and B. Juang
'''

from mpi4py import MPI

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from sys import argv
from argparse import ArgumentParser, Namespace

from OFDM_funtions import *


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
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-snr", "--SNR", type=float, default=30)
    parser.add_argument("-wd", "--weight_decay", type=float, default=2e-3)
    parser.add_argument("-p", "--pilots", type=int, default=64)
    parser.add_argument("-t", "--trials", type=int, default=10)
    return parser.parse_args(argv[1:])

## Client netowrk architecture
class ClientNetwork(nn.Module):

    def __init__(self):
        super(ClientNetwork, self).__init__()
        self.linear = nn.Linear(256,500)
        self.linear1 = nn.Linear(500, 250)

    
    def forward(self, x):        
        x = F.relu(self.linear(x))
        x = F.relu(self.linear1(x))
        return x 

## Server network architecture
class ServerNetwork(nn.Module):

    def __init__(self):
        super(ServerNetwork, self).__init__()
        self.linear2 = nn.Linear(250, 120)
        self.linear3 = nn.Linear(120, 16)
    
    
    def forward(self, x):
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

## Function to apply the simulated channel vectors to the input OFDM symbols
def data_gen(channel_data, SNRdb, dset):
    ## Setting dataset sizes
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
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))  # create random data carrying symbols
        signal_output, sent_symbols = ofdm_simulate(bits, H, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers)  # generate model inputs and labels
        input_labels[i, :] = sent_symbols
        input_samples[i, :] = signal_output
    return torch.utils.data.TensorDataset(torch.from_numpy(input_samples), torch.from_numpy(input_labels))  # Create torch dataset


def create_dataset(path, train_or_test, snr):
    data = np.load(path)  # Load data from file

    ## Create data loaders for torch 
    if train_or_test == 'train':
        received = data_gen(data, snr, dset=train_or_test)  # Function that simulates the effect of a channel on the input symbols
        return DataLoader(received, batch_size=args.batch_size, shuffle=True)
    elif train_or_test == 'val':
        received = data_gen(data, snr, dset=train_or_test)  # Function that simulates the effect of a channel on the input symbols
        return DataLoader(received, batch_size=args.val_batch_size, shuffle=True)
    elif train_or_test == 'test':
        received = data_gen(data, snr, dset=train_or_test)  # Function that simulates the effect of a channel on the input symbols
        return DataLoader(received, batch_size=args.test_batch_size, shuffle=True)

## Function to calculuate the bit error from the model predicted I and Q values
def bit_error(y_true, y_pred):
    y_pred_sym = torch.sign(y_pred)  # Find the closest QPSK symbol to predictions 
    return 1 - torch.mean(torch.mean(torch.eq(y_true, y_pred_sym).type(torch.float32),1)).item()  # Find the amount of I and Q errors

## Function for testing the trained model differnt SNR values
def SNR_testing():

    path = "../saved_models/"

    ## intilize models 
    client_model = ClientNetwork()
    server_model = ServerNetwork()

    ## Load trained parameters into models
    client_model.load_state_dict(torch.load(path + "client_model_" + str(P) + "P.pt"))
    server_model.load_state_dict(torch.load(path + "server_model_" + str(P) + "P.pt"))

    ## Set models to evaluate
    client_model.eval()
    server_model.eval()

    loss_func = nn.MSELoss()  # Define loss function
    
    for snr in SNRs:
        total_loss = 0.0
        BER = 0.0 
        steps = 0
        
        test_data = create_dataset("../data/channel_test_full.npy", "test", snr)
        
        ## Test pass
        for input, labels in test_data:
            with torch.no_grad():
                cut_tensor = client_model(input)
                output = server_model(cut_tensor)
            loss = loss_func(labels, output)
            total_loss += loss.item()
            BER += bit_error(labels, output)
            steps +=1
        
        BER /= steps

        results[snr] += BER / trials


def split_learning():
    ## Set the state of the curent process as a worker or the server
    if rank >= 1:
        worker, server = True, False
    else:
        server, worker = True, False

    ## Code block run on each worker process for training and evalutation
    if worker:
        
        epoch = 1
        active_worker = rank  # Sets active worker as current process

        if MAX_RANK > 1:

            ## Sets the position in the worker queue
            worker_left = rank - 1
            worker_right = rank + 1

            if rank == 1:
                worker_left = MAX_RANK  # Wrapping the worker queue so its a loop and skipping worker 0, i.e. the server

            ## Make sure that worker rank 1 is the first to start
            elif rank == MAX_RANK:
                worker_right = 1  # Wrapping the worker queue so its a loop and skipping worker 0, i.e. the server
                comm.send("start", dest=worker_right)  # Wend start message to the worker to the right to begin training on that process

        ## Creating datasets using the channel vectors and QPSK symbols
        train = create_dataset("../data/channel_train_full.npy", "train", SNR)
        if rank == MAX_RANK:
            validation = create_dataset("../data/channel_test_full.npy", "val", SNR)
            test = create_dataset("../data/channel_test_full.npy", "test", SNR)

        client_model = ClientNetwork().to(device)   # Initialize the client network to device
        client_optimizer = optim.Adam(client_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # Initialize the client optimizer
        client_scheduler = optim.lr_scheduler.StepLR(client_optimizer, step_size=10, gamma=0.4)   # Reduce learning rate after 10 epochs using scheduler

        while True:
            # Wait to receive a message from the other worker
            if MAX_RANK > 1:
                msg = comm.recv(source=worker_left)
            else:
                msg = "start"

            if msg == "start":

                client_model.train() 

                for inputs, labels in train:
                    inputs, labels = inputs.to(device), labels.to(device)  # Send the input data and lables to device
                    client_optimizer.zero_grad()  # Clear optimizer grads
                    split_layer_tensor = client_model(inputs)  # Forward pass
                    comm.send(["tensor_and_labels", [split_layer_tensor, labels]], dest=SERVER)  # Send the cut layer tensor and the labels to the server
                    grads = comm.recv(source=SERVER)  # Receive the gradients for backpropagation from the server
                    split_layer_tensor.backward(grads)  # Apply the gradients at the cut layer
                    client_optimizer.step()  # Apply the optimizer

                del split_layer_tensor, grads, inputs, labels  # Delete used tensors ready for next epoch on this worker
                torch.cuda.empty_cache()

                # Only the last worker evaluates on the validation data
                if rank == MAX_RANK and epoch % 10 == 0:
                    comm.send("validation", dest=SERVER)  # Tell the server to go into validation phase
                    client_model.eval()  # Client model in evaluation mode 

                    for inputs, labels in validation:
                        inputs, labels = inputs.to(device), labels.to(device)  # Send input data and lables to device
                        with torch.no_grad():
                            split_layer_tensor = client_model(inputs)  # Forward pass
                        comm.send(["tensor_and_labels", [split_layer_tensor, labels]], dest=SERVER)  # Send the cut layer tensor and the labels to the server

                    del split_layer_tensor, inputs, labels  # Delete used tensors ready for next epoch on this worker
                    torch.cuda.empty_cache()

                if rank == MAX_RANK and epoch == args.epochs:
                    comm.send("test", dest=SERVER)  # Tell the server to go into test phase
                    client_model.eval()  # Client model in evaluation mode 

                    for inputs, labels in test:
                        inputs, labels = inputs.to(device), labels.to(device)  # Send input data and lables to device
                        with torch.no_grad():
                            split_layer_tensor = client_model(inputs)  # Forward pass
                        comm.send(["tensor_and_labels", [split_layer_tensor, labels]], dest=SERVER)  # Send the cut layer tensor and the labels to the server

                    del split_layer_tensor, inputs, labels  # Delete used tensors ready for next epoch on this worker
                    torch.cuda.empty_cache()

                if MAX_RANK > 1:
                    comm.send("start", dest=worker_right)  # Signal to the next worker to start training on its dataset

                if epoch == args.epochs:
                    # Let the server know each worker has fininshed the last epoch and stop each while loop 
                    msg = "training_complete" if rank == MAX_RANK else "worker_done"
                    comm.send(msg, dest=SERVER)
                    if rank == MAX_RANK:
                        torch.save(client_model.state_dict(), "../saved_models/client_model_"+ str(P) + "P.pt")  # save model for testing
                    break
                else:
                    # Let the server know that the current epoch has finished
                    msg = "epoch_done" if rank == MAX_RANK else "worker_done"
                    comm.send(msg, dest=SERVER)
                epoch += 1
                client_scheduler.step()
    
    # Define the procedure for the server
    elif server:
        server_model = ServerNetwork().to(device)  # Initialize the server network to device
        loss_func = nn.MSELoss()  # Loss function
        server_optimizer = optim.Adam(server_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # Initialize the server optimizer
        server_scheduler = optim.lr_scheduler.StepLR(server_optimizer, step_size=10, gamma=0.4)  # Reduce learning rate after 10 epochs

        ## Initialize train, test, validation metrics
        train_loss, train_step, train_bit_error = 0.0, 0, 0.0
        val_loss, val_step, val_bit_error  = 0.0, 0, 0.0 
        test_loss, test_step, test_bit_error = 0.0, 0, 0.0
        epoch = 1
        active_worker, phase = 1, "train"

        while(True):
            msg = comm.recv(source=active_worker)  # Get message from the active worker

            if msg[0] == "tensor_and_labels":

                split_layer_tensor, labels = msg[1]  # Retrieve the input and labels from the message sent by the active worker
                
                if phase == "train":

                    server_model.train()  # Set server to train mode
                    server_optimizer.zero_grad()  # Clear optimizer grads
                    outputs = server_model(split_layer_tensor)  # Forward pass
                    loss = loss_func(outputs, labels)  # Loss calculation
                    
                    ## Update train metrics
                    train_loss += loss.item()
                    train_bit_error += bit_error(labels, outputs)
                    train_step += 1

                    loss.backward()  # Back propagation
                    server_optimizer.step()  # Apply the optimizer
                    comm.send(split_layer_tensor.grad, dest=active_worker)  # Send gradients back to the active worker
        
                if phase == "validation":

                    server_model.eval()  # Set model to evaluation mode

                    with torch.no_grad():
                        outputs = server_model(split_layer_tensor)  # Forward pass

                    loss = loss_func(outputs, labels)  # Loss calculation
                    
                    ## Update validation metrics
                    val_bit_error += bit_error(labels, outputs)
                    val_loss += loss.item()
                    val_step += 1
                
                if phase == "test":
                
                    server_model.eval()  # Set model to evaluation mode

                    with torch.no_grad():
                        outputs = server_model(split_layer_tensor)  # Forward pass

                    loss = loss_func(outputs, labels)  # Loss Calculation
                    
                    ## Update test metrics
                    test_loss += loss.item()
                    test_bit_error += bit_error(labels, outputs)
                    test_step += 1

            elif msg == "worker_done":
                if MAX_RANK > 1:
                    # Change worker and phase
                    active_worker = (active_worker % MAX_RANK) + 1
                    phase = "train"

            elif msg == "epoch_done" or msg == "training_complete":

                server_scheduler.step()  # Apply scheduler step 

                if epoch % 10 == 0:
                    
                    ## Calculate train and validation metrics for that epoch
                    train_loss /= train_step
                    train_bit_error /= train_step
                    val_loss /= val_step
                    val_bit_error /= val_step

                    print("Epoch {} - Train: loss {}, bit error {} -  Val: loss {}, bit error {}\n".format(epoch, train_loss, 
                                                                                                            train_bit_error, val_loss, 
                                                                                                            val_bit_error))

                ## Set next epoch
                if active_worker == MAX_RANK:
                    epoch += 1

                ## Change worker and phase
                active_worker = (active_worker % MAX_RANK) + 1
                phase = "train"

                if msg == "training_complete":

                    torch.save(server_model.state_dict(), "../saved_models/server_model_"+ str(P) + "P.pt")  # save model for SNR testing

                    ## Calculate test metrics after training complete
                    test_loss /= test_step
                    test_bit_error /= test_step

                    print("Test loss: {:.4f}".format(test_loss))
                    print("Test bit error: {:.4f}\n".format(test_bit_error))
                    break
                
                ## Reset validation and train metric ready for next epoch
                train_loss, train_bit_error, train_step = 0.0, 0.0, 0
                val_loss, val_bit_error = 0.0, 0.0

            elif msg == "validation":
                # Change phase and reset variables
                phase = "validation"
                val_step = 0
            
            elif msg == "test":
                # Change phase and reset variables
                phase = "test"
                test_step = 0
    
                
if __name__ == '__main__':

    args = parse_args()

    # Define the communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # code is run on n parrallel virtual machines, this sets the rank on each so for worker 1, rank = 1

    SERVER = 0  # set server rank as 0, all workers send their cut layer tensor to this worker to complete forward pass
    MAX_RANK = comm.Get_size() - 1  # max rank set to number of client workers, so will be one less than total workers

    # choose to run on cpu or cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    SNR = args.SNR
    BER = 0.0
    K = 64  # number of sub-carriers per OFDM symbol
    CP = K//4  # cyclic prefix
    P = args.pilots # number of pilot symbols
    allCarriers = np.arange(K)  # indices of subcarriers
    mu = 2  # no of OFDM symbols in a frame
    CP_flag = True  # Choose to include a cyclic prefix or not

    trials = args.trials

    if P < K:
        pilotCarriers = allCarriers[::K//P]  # pilots are every (K/P)th carrier if there are less that K pilots
        dataCarriers = np.delete(allCarriers, pilotCarriers)  # delete pilots from the data carrier vector
    else:
        pilotCarriers = allCarriers  # all sub-carriers hold a piot symbol in the first OFDM symbol
        dataCarriers = [] 

    payloadBits_per_OFDM = K * mu  # total number of sub-carrier symbols in the frame
    
    ## load pilots from txt file or create a new one with random bits that are kept constant in training and testing
    Pilot_file_name = 'Pilot_'+str(P)
    if os.path.isfile(Pilot_file_name):
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        np.savetxt(Pilot_file_name, bits, delimiter=',')

    pilotValue = Modulation(bits, mu)  # create QPSK symbols for the pilot bits

    SNRs = [5,10,15,20,25]  # SNR values to be tested
    results = {snr: 0.0 for snr in SNRs}  # Results dict

    for _ in range(trials):
        split_learning()
        if rank == 0:
            SNR_testing()

    if rank == 0:
        print("Results with {} pilots:\n {}".format(P, results))



    