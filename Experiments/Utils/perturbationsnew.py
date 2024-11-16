
from scipy import signal
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def multisamplebackgroundIdentification(global_signals):
    test_sig = global_signals[0].flatten()
    f, t, Zxx = signal.stft(test_sig,1,nperseg=40)
    for i in range(1,len(global_signals)):
        test_sig = global_signals[i].flatten()
        f0, t0, Zxx0 = signal.stft(test_sig,1,nperseg=40, noverlap = 20)
        assert Zxx.shape == Zxx0.shape, f"Shape mismatch at signal {i}"
        Zxx = Zxx + Zxx0
    frequency_composition_abs = np.abs(Zxx)
    measures = []
    for freq,freq_composition in zip(f,frequency_composition_abs):
        measures.append(np.mean(freq_composition)/np.std(freq_composition))
    max_value = max(measures)
    selected_frequency = measures.index(max_value)
    weights = 1-(measures/sum(measures))
    dummymatrix = np.zeros((len(f),len(t)))
    dummymatrix[selected_frequency,:] = 1  
    #Option to admit information from other frequency bands
    """dummymatrix = np.ones((len(f),len(t)))
    for i in range(0,len(weights)):
        dummymatrix[i,:] = dummymatrix[i,:] * weights[i]"""
    
    background_frequency = Zxx * dummymatrix / global_signals.shape[0]
    _, xrec = signal.istft(background_frequency, 1)
    return xrec



def backgroundfreqIdentification(original_signal):
    f, t, Zxx = signal.stft(original_signal,fs = 1,nperseg=40, noverlap = 20)
    frequency_composition_abs = np.abs(Zxx)
    measures = []
    for freq,freq_composition in zip(f,frequency_composition_abs):
        measures.append(np.mean(freq_composition)/np.std(freq_composition))
    max_value = max(measures)
    selected_frequency = measures.index(max_value)
    Zxx_candidate = np.zeros_like(Zxx, dtype=complex)
    # insert the selected component into selected frequency
    Zxx_candidate[selected_frequency, :] = Zxx[selected_frequency, :]
    return Zxx_candidate



#Background Identification
def candidatebackgroundIdentification(global_signals):
    test_sig = global_signals[0].flatten()
    f_, t_, Zxx_ = signal.stft(test_sig,1,nperseg=40)
    Zxx = np.zeros_like(Zxx_, dtype=complex)

    #For every sample, return a spectral matrix representing the background 
    for i in range(1,len(global_signals)):
        test_sig = global_signals[i].flatten()
        Zxx_candidate = backgroundfreqIdentification(test_sig)
        assert Zxx.shape == Zxx_candidate.shape, f"Shape mismatch at signal {i}"
        #Add all backgrounds together
        Zxx = Zxx + Zxx_candidate
    frequency_composition_abs = np.abs(Zxx)
    measures = []
    for freq,freq_composition in zip(f_,frequency_composition_abs):
        measures.append(np.mean(freq_composition)/np.std(freq_composition))
    max_value = max(measures)
    selected_frequency = measures.index(max_value)
    weights = 1-(measures/sum(measures))
    dummymatrix = np.zeros((len(f_),len(t_)))
    dummymatrix[selected_frequency,:] = 1  
    
    background_frequency = Zxx * dummymatrix
    _, xrec = signal.istft(background_frequency, 1)
    return xrec
    

def backgroundIdentification(original_signal):
    f, t, Zxx = signal.stft(original_signal,1,nperseg=40)
    frequency_composition_abs = np.abs(Zxx)
    measures = []
    for freq,freq_composition in zip(f,frequency_composition_abs):
        measures.append(np.mean(freq_composition)/np.std(freq_composition))
    max_value = max(measures)
    selected_frequency = measures.index(max_value)
    weights = 1-(measures/sum(measures))
    dummymatrix = np.zeros((len(f),len(t)))
    dummymatrix[selected_frequency,:] = 1  
    #Option to admit information from other frequency bands
    """dummymatrix = np.ones((len(f),len(t)))
    for i in range(0,len(weights)):
        dummymatrix[i,:] = dummymatrix[i,:] * weights[i]"""
    
    background_frequency = Zxx * dummymatrix
    _, xrec = signal.istft(background_frequency, 1)
    return xrec

def RBP(generated_samples_interpretable, original_signal, segment_indexes):
    generated_samples_raw = []
    xrec = backgroundIdentification(original_signal)
    for sample_interpretable in generated_samples_interpretable:
        raw_signal = original_signal.copy()
        for index in range(0,len(sample_interpretable)-1):
            if sample_interpretable[index] == 0:
                index0 = segment_indexes[index]
                index1 = segment_indexes[index+1]
                raw_signal[index0:index1] = xrec[index0:index1]
        generated_samples_raw.append(np.asarray(raw_signal))
    return np.asarray(generated_samples_raw)

def RBPIndividual(original_signal, index0, index1):
    xrec = backgroundIdentification(original_signal)
    raw_signal = original_signal.copy()
    raw_signal[index0:index1] = xrec[index0:index1]
    return raw_signal


def RBPIndividualNew1(global_signals, original_signal, index0, index1):
    xrec = multisamplebackgroundIdentification(global_signals)
    raw_signal = original_signal.copy()
    raw_signal[index0:index1] = xrec[index0:index1]
    return raw_signal


def RBPIndividualNew2(global_signals, original_signal, index0, index1):
    xrec = candidatebackgroundIdentification(global_signals)
    raw_signal = original_signal.copy()
    raw_signal[index0:index1] = xrec[index0:index1]
    return raw_signal


def zeroPerturb(original_signal, index0, index1):
    new_signal = original_signal.copy()
    new_signal[index0:index1] = np.zeros(index1 - index0)
    return new_signal

def noisePerturb(original_signal, index0, index1):
    new_signal = original_signal.copy()
    new_signal[index0:index1] = np.random.randint(np.min(original_signal),np.max(original_signal),index1 - index0).reshape(index1 - index0)
    return new_signal 


def blurPerturb(original_signal, index0, index1):
    new_signal = original_signal.copy()
    new_signal[index0:index1] = gaussian_filter(new_signal[index0:index1],np.std(original_signal))
    return new_signal

