This Repository contains the supporting material for the paper, "Distributed Learning of Universal Recievers Over Fading Channels in IoT Communications". 

The scripts contain elements taken from the the support code of various papers. The split learning algorithm is adapted from [1], and the federated learning algorithm is adapted from [2]. There are also section adpated from [2] for channel modelling and [3] for the channel modelling in the OFDM case. 

The Repository also contains the datasets taken from [2].

The scripts have been tested on Python 3.9.7 run on Ubuntu 20.04

The scripts SplitRec.py and OFDM_split.py require the use of an MPI. This is sepeartely installed program. The the MPI used here is Open MPI, the download and installation instructions are found here: https://www.open-mpi.org/software/ompi/v4.1/

Requirements:

all:
numpy=~1.21.2

OFDM_split.py and SplitRec.py:
scikit-learn=~1.0
pytorch=~1.9.0
mpi4py=~3.1.1

OFDM_federated.py:
tensorflow=~2.5.1
tensorflow-federated=~0.19.0

To run OFDM_split.py and SplitRec.py with 5 worker devices

$ mpiexec -np 6 python src/"file_name".py

To run OFDM _federated.py

$ python src/"file_name.py"

The training test data for the SISO reciever is included in this repository. The channel datasets for the OFDM reciever are too large and are availible for dowload from the orginal GitHub repo: https://github.com/haoyye/OFDM_DNN. 

Any command line arguments that adjust simulation paramaters are found in the scripts.

[1] Ioannidou X, Straathof, B, T., (2020) Scalable Machine Learning ID2223
Project: An Investigation into Split Learning

[2] M. B. Mashhadi, N. Shlezinger, Y. C. Eldar, and D. Gunduz, “Fe-
drec: Federated learning of universal receivers over fading channels,”
arXiv:2011.07271, 2021

[3] H. Ye, G. Y. Li, and B. H. Juang, “Power of Deep Learning for Channel
Estimation and Signal Detection in OFDM Systems,” IEEE Wireless
Commun. Lett., vol. 7, no. 1, Feb. 2018, pp. 114–17.
