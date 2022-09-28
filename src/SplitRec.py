'''SplitRec.py uses a modified version of the split learning implementation from:
"Scalable Machine Learning ID2223 Project: An Investigation Into Split Learning" - X. Ioannidou, B. T. Straathof.
Dataset generation code adpated from the GitHub repsository: https://github.com/MahdiBoloursazMashhadi/FedRec for the paper:
"FedRec: Federated Learning of Universal Receivers over Fading Channels" - M., B. Mashhadi et al.
'''

from mpi4py import MPI

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from sys import argv
from argparse import ArgumentParser, Namespace

from sklearn.model_selection import train_test_split

def parse_args() -> Namespace:
    """Parses CL arguments

    Returns:
        Namespace object containing all arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-bs", "--batch_size", type=int, default=200)
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=100000)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--epochs", type=int, default=31)
    parser.add_argument("-t", '--trials', type=int, default=100)
    parser.add_argument("-snr", "--SNR", type=float, default=5)
    parser.add_argument("-iid", "--IID", type=bool)
    parser.add_argument("-stn", "--split_tensor_nodes", type=int, default=32)

    return parser.parse_args(argv[1:])

## Training data generation
def traingen(x, y):
    if iid:
        sigma = 1  # i.i.d. user fading
    else:
        sigma = np.random.uniform(low=0.5, high=1.5, size=None)  # non-i.i.d. user fading

    h = np.random.rayleigh(scale=sigma, size=(x.shape[0], 1))
    hIQ = np.concatenate((h, h), axis=1)  # Creates matrix of channel coefficients
    x_u = np.multiply(hIQ, x) + np.random.normal(0, np.sqrt(N0)/2, x.shape)  # Distorting symbols with channel and AWGN 
    x_u = torch.from_numpy(x_u).float()
    y_u = torch.from_numpy(y).long()
    d_u = torch.utils.data.TensorDataset(x_u, y_u)  # Creating torch dataset object
    return d_u

## Test/val data generation
def testgen(x, y):
    if iid==True:
        h = np.random.rayleigh(scale=1, size=(x.shape[0],1))  # i.i.d. test fading 
    else:
        h = np.random.rayleigh(scale=np.random.uniform(low=0.5, high=1.5, size=(x.shape[0],1)), size=(x.shape[0],1))  # non-i.i.d. user fading
    
    hIQ = np.concatenate((h, h), axis=1)  # Creates matrix of channel coefficients
    x_u = np.multiply(hIQ, x) + np.random.normal(0, np.sqrt(N0)/2, x.shape)  # Distorting symbols with channel and AWGN 
    x_u = torch.from_numpy(x_u).float()
    y_u = torch.from_numpy(y).long()
    return x_u, y_u

## Function that creates the data sets on each worker
def create_dataset(path, train_or_test):

    data = np.load(path)  # Load data from file

    ## Create data loaders for torch 
    if train_or_test == 'train':
        start = int(np.floor(len(data)/MAX_RANK*(rank-1)))  # Lower limit for splitting training data across workers
        stop = int(np.floor(len(data)/MAX_RANK*(rank)))  # Upper limit for splitting training data across workers
        data = data[start:stop]  # Splits input data across workers
        received = traingen(data[:, 0:2], data[:, 2])  # Simulates the effect of a channel on the input symbols
        return DataLoader(received, batch_size=args.batch_size, shuffle=True)
    else:
        x_u, y_u = testgen(data[:, 0:2], data[:, 2])  # simulates the effect of a channel on the input symbols
        x_test, x_val, y_test, y_val = train_test_split(x_u, y_u, test_size=0.05, shuffle=True)
        val = torch.utils.data.TensorDataset(x_val, y_val)
        test = torch.utils.data.TensorDataset(x_test, y_test)
        return DataLoader(val, batch_size=args.test_batch_size, shuffle=True), DataLoader(test, batch_size=args.test_batch_size, shuffle=True)


## Class defining the part of the network to run on each client worker
class ClientNetwork(nn.Module):

    def __init__(self):
        super(ClientNetwork, self).__init__()
        self.linear1 = nn.Linear(2, 16)
        self.linear2 = nn.Linear(16, args.split_tensor_nodes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x 


## Class defining the part of the network on the server worker
class ServerNetwork(nn.Module):

    def __init__(self):
        super(ServerNetwork, self).__init__()
        self.linear1 = nn.Linear(args.split_tensor_nodes, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def split_learning():
    ## Set the state of the current process as a worker or the server
    if rank >= 1:
        worker, server = True, False
    else:
        server, worker = True, False

    ## Code run on each worker process
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
                comm.send("start", dest=worker_right)  # Send start message to the worker to the right to beign training on that process

        train_loader = create_dataset('../data/train_full.npy', 'train')

        if rank == MAX_RANK:
            val_loader, test_loader = create_dataset('../data/test.npy', 'test')
            
        client_model = ClientNetwork().to(device)
        client_optimizer = optim.Adam(client_model.parameters(), lr=args.learning_rate)

        while True:
            ## Wait to receive a message from the other worker
            if MAX_RANK > 1:
                msg = comm.recv(source=worker_left)
            else:
                msg = "start"

            if msg == "start":
                client_model.train() 

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)  # Send input data and lables to cpu or cuda
                    client_optimizer.zero_grad()  # Clear optimizer grads
                    cut_layer_tensor = client_model(inputs)  # Client model forward pass
                    comm.send(["tensor_and_labels", [cut_layer_tensor, labels]], dest=SERVER)  # Send the cut layer tensor and the labels to the server
                    grads = comm.recv(source=SERVER)  # Receive the gradients for backpropgation from the server
                    cut_layer_tensor.backward(grads)  # Apply the gradients to the cut layer
                    client_optimizer.step()  # Apply optimizer step

                del cut_layer_tensor, grads, inputs, labels  # Delete used tensors ready for next epoch on this worker
                torch.cuda.empty_cache()  

                if rank == MAX_RANK and epoch % 10 == 0:
                    comm.send("validation", dest=SERVER)  # Tell server to go into validation phase
                    client_model.eval()  # Client model in evaluation mode 

                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)  # Send input data and lables to cpu or cuda
                        with torch.no_grad():
                            cut_layer_tensor = client_model(inputs)  # Forward pass
                        comm.send(["tensor_and_labels", [cut_layer_tensor, labels]], dest=SERVER)  # Send the cut layer tensor and the labels to the server

                    del cut_layer_tensor, inputs, labels  # delete used tensors ready for next epoch on this worker
                    torch.cuda.empty_cache()

                if rank == MAX_RANK and epoch == args.epochs:
                    comm.send("test", dest=SERVER)  # Tell server to go into test mode
                    client_model.eval()  # Client model in evaluation mode 

                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)  # Send input data and lables to cpu or cuda
                        with torch.no_grad():
                            cut_layer_tensor = client_model(inputs)  # Forward pass
                        comm.send(["tensor_and_labels", [cut_layer_tensor, labels]], dest=SERVER)  # Send the cut layer tensor and the labels to the server

                    del cut_layer_tensor, inputs, labels  # Delete used tensors ready for next epoch on this worker
                    torch.cuda.empty_cache()

                if MAX_RANK > 1:
                    comm.send("start", dest=worker_right)  # Signal to the next worker to start training

                if epoch == args.epochs:
                    ## Let the server know each worker has fininshed the last epoch and stop each while True loop 
                    msg = "training_complete" if rank == MAX_RANK else "worker_done"
                    comm.send(msg, dest=SERVER)
                    break
                else:
                    ## Let the server know that the current epoch has finished
                    msg = "epoch_done" if rank == MAX_RANK else "worker_done"
                    comm.send(msg, dest=SERVER)
                epoch += 1
    
    ## Procedure for the server
    elif server:
        server_model = ServerNetwork().to(device)  # Initialize the server network to cpu or cuda
        loss_func = nn.CrossEntropyLoss()  # Define the loss function
        server_optimizer = optim.Adam(server_model.parameters(), lr=args.learning_rate)  # Initialize the server optimizer

        active_worker, phase = 1, "train"
        train_loss, train_step = 0.0, 0
        val_loss, val_step = 0.0, 0 
        test_loss, test_step = 0.0, 0
        epoch = 0 
        total_n_labels_train, correct_train = 0, 0
        total_n_labels_val, correct_val = 0, 0
        total_n_labels_test, correct_test = 0, 0
        
        while(True):
            msg = comm.recv(source=active_worker)  # Get message from the active worker

            if msg[0] == "tensor_and_labels":
                cut_layer_tensor, labels = msg[1]  # Retrieve the input and labels from the message sent by the active worker
                
                if phase == "train":
                    server_model.train()  # Set server model to train mode
                    server_optimizer.zero_grad()  # Clear optimizer grad
                    outputs = server_model(cut_layer_tensor)  # Forward pass through server model
                    loss = loss_func(outputs, labels)  # Loss calculation

                    ## Train metrics
                    total_n_labels_train += len(labels)  # Add current label count to the total number of labels
                    _, predictions = outputs.max(1)  # Find most probable label from softmax layer
                    correct_train += predictions.eq(labels).sum().item()  # Identify correct predictions
                    train_loss += loss.item()
                    train_step += 1  # Keep track of batches

                    loss.backward()  # Back propagation
                    server_optimizer.step()  # Apply the optimizer
                    comm.send(cut_layer_tensor.grad, dest=active_worker)  # Send gradients back to the active worker

                if phase == "validation":
                    server_model.eval()  # Set model to evaluation mode

                    with torch.no_grad():
                        outputs = server_model(cut_layer_tensor)  # Forward pass through server model with no grad

                    ## Validation metrics
                    total_n_labels_val += len(labels)  # Add current label count to the total number of labels
                    _, predictions = outputs.max(1)  # Find most probable label from softmax layer
                    loss = loss_func(outputs, labels)  # Loss calculation
                    correct_val += predictions.eq(labels).sum().item()  # Identify how many of the predictions were correct
                    val_loss += loss.item() 
                    val_step += 1  # Keep track of batches

                if phase == "test":
                    server_model.eval()  # Set model to evaluation mode

                    with torch.no_grad():
                        outputs = server_model(cut_layer_tensor)  # Forward pass through server model with no grad

                    ## Test metrics
                    total_n_labels_test += len(labels)  # Add current label count to the total number of labels
                    _, predictions = outputs.max(1)  # Find most probable label from softmax layer
                    loss = loss_func(outputs, labels)  # Loss calculation
                    correct_test += predictions.eq(labels).sum().item()  # Identify how many of the predictions were correct
                    test_loss += loss.item() 
                    test_step += 1  # Keep track of batches

            elif msg == "worker_done":
                # Change worker and phase
                if MAX_RANK > 1:
                    active_worker = (active_worker % MAX_RANK) + 1  # sets next active worker and goes back to worker 1 when MAX_RANK is reached
                    if epoch < args.epochs:
                        phase = "train"

            elif msg == "epoch_done" or msg == "training_complete":

                if epoch % 10 == 0:
                    ## For keeping track of validation or train metrics
                    train_acc = correct_train / total_n_labels_train
                    train_loss /= train_step

                    val_acc = correct_val / total_n_labels_val
                    val_loss /= val_step

                    print("Epoch {} - Train: loss {}, accuracy {} -  Val: loss {}, accuracy {}\n".format(epoch, train_loss, 
                                                                                                        train_acc, val_loss, 
                                                                                                        val_acc))

                #  set the next epoch
                if active_worker == MAX_RANK:
                    epoch += 1  

                # Change worker and phase
                active_worker = (active_worker % MAX_RANK) + 1
                phase = "train"

                if msg == "training_complete":
                    test_loss /= test_step   # Calculate the average test loss across users
                    test_acc = correct_test / total_n_labels_test  # Calculate the average test accuracy across users
                    BER = ((1-test_acc)/(Q*trials))  # calculate average BER over all trials 
                    print("{} trials complete: acc - {} loss - {} BER - {}\n".format(m + 1, test_acc, test_loss, BER*trials))
                    return BER
                
                ## Resetting validation and train variable between epochs
                total_n_labels_val, correct_val = 0, 0
                total_n_labels_train, correct_train = 0, 0
                train_loss, train_step = 0.0, 0
                val_loss, val_step = 0.0,  0
        
            elif msg == "test":
                phase = "test"  # Change phase
            
            elif msg == "validation":
                phase = "validation"  # Change phase
    
                
if __name__ == '__main__':

    args = parse_args()  # parse command line arguements

    comm = MPI.COMM_WORLD  # define the communicator
    rank = comm.Get_rank() # code is run on parrallel processes, this sets the rank on each so for worker 1, rank = 1

    SERVER = 0  # set server rank as 0, all workers send their cut layer tensor to this worker to complete forward pass
    MAX_RANK = comm.Get_size() - 1  # max rank set to number of client workers, so will be one less than total workers

    torch.manual_seed(0)

    # Choose to run of cuda or cpu, with small model size, its quicker run on cpu 
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    trials = args.trials # Number of Monte Carlo trials
    iid = args.IID  # set to False to simulate non-iid user fading

    #  Modulation Parameters
    Q = 4
    M = 2 ** Q  # 16QAM modulation
    Es = 10  # Average symbol energy

    #  Noise Parameters
    EbN0_dB = args.SNR # SNR per bit (dB)
    EsN0_dB = EbN0_dB + 10 * np.log10(Q)  # SNR per symbol (dB)
    N0 = Es / 10 ** (EsN0_dB / 10)

    BER = 0.0
    for m in range(trials):  # run the split learning for each trial
        results = split_learning()
        if rank == 0:
            BER += results  # accumulating the average BER over all trials
    
    if rank == 0:
        print('##############################')
        print('16QAM at Eb/N0 =', EbN0_dB, 'dB')
        print(MAX_RANK, 'users on', device)
        iidstr = 'iid' if iid else 'non-iid'
        print(iidstr, 'Rayleigh fading')
        print('BER =', BER)
        print('##############################')

