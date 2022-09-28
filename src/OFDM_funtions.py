'''Functions taken from the GitHub repository: https://github.com/haoyye/OFDM_DNN for the paper:
"Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems" - H. Ye , G. Ye Li, and B. Juang
These function are used in the training scripts to generate the datasets'''

import numpy as np

## Converts bit stream into QPSK modulated symbols 
def Modulation(bits, mu):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return (2 * bit_r[:, 0] - 1) + 1j * (2 * bit_r[:, 1] - 1)

## Adds the cyclic prefix to the OFDM symbols in the frame
def addCP(OFDM_time, CP_flag, CP, mu, K):
    
    if CP_flag == False:
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K*mu, ))  # Create random bits
        symbols_noise = Modulation(bits_noise, mu)  # To symbols
        OFDM_time_noise = np.fft.ifft(symbols_noise)  # Time domain
        cp = OFDM_time_noise[-CP:]  # Take noisey signal to create CP, for simulating there being no CP 
    else:
        cp = OFDM_time[-CP:]  # take the last CP samples
    return np.hstack([cp, OFDM_time])  # add them to the beginning

## Applies channel effect to an OFDM symbol
def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)  # Convolves the time domain signal with the channel vector
    signal_power = np.mean(abs(convolved**2))  #  Extract signal power as the signal amplitude squared
    noise_power = signal_power * 10**(-SNRdb / 10)  #  Calculate noise power 
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))  # AWGN on the signal with variance of noise_power
    return convolved + noise  # Return final signal, peturbed by AWGN and 

## Simulates the transmission of an OFDM frame across a sample spaced multipath channel
def ofdm_simulate(codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,):
    bits = np.random.binomial(n=1, p=0.5, size=(2*(K - P),))  # Generates random bits for the first pilot containing OFDM frame if P < K
    QPSK = Modulation(bits, mu)  # create QPSK symbols
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[pilotCarriers] = pilotValue  # Assigns pilot symbols to pilot sub-carriers
    OFDM_data[dataCarriers] = QPSK  # Assigns data symbols to data subcarriers (if P < K)
    OFDM_time = np.fft.ifft(OFDM_data)  # Transfrom into time domain signal 
    OFDM_withCP = addCP(OFDM_time, CP_flag, CP, mu, K)  # Add cyclic prefix (noise prefix if no cp used)
    OFDM_RX = channel(OFDM_withCP, channelResponse, SNRdb)  # Simulate channel transmission
    OFDM_RX_noCP = OFDM_RX[CP:(CP + K)]  # Remove cyclic prefix
    OFDM_RX_noCP = np.fft.fft(OFDM_RX_noCP)  # Transform back into frequency domain to retrieve sub-carrier symbols

    ## Same process for the data carrying OFDM symbol
    codeword_qam = Modulation(codeword, mu)
    OFDM_time_codeword = np.fft.ifft(codeword_qam)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP_flag, CP, mu, K)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = OFDM_RX_codeword[CP:(CP + K)]
    OFDM_RX_noCP_codeword = np.fft.fft((OFDM_RX_noCP_codeword))
    
    ## Create imput data by splitting the real and imaginary parts of the sub-carrier symbols 
    inputs = np.zeros((256,))
    inputs[0:inputs.shape[0]:2] = np.concatenate((np.real(OFDM_RX_noCP), np.real(OFDM_RX_noCP_codeword)))
    inputs[1:inputs.shape[0]:2] = np.concatenate((np.imag(OFDM_RX_noCP), np.imag(OFDM_RX_noCP_codeword)))

    ## Create labels by extracting the I and Q values of the first 8 sub-carrier symbols in the data carrying OFDM symbol
    labels = np.zeros((16,))
    labels[0:labels.shape[0]:2] = np.real(codeword_qam[0:8])
    labels[1:labels.shape[0]:2] = np.imag(codeword_qam[0:8])
    return inputs, labels