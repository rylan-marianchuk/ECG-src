import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import torch
from torch.utils.data import Dataset

from transforms import Wavelet


class SinusoidDataSet(Dataset):

    def __init__(self, size, sinusoids, max_freq=15.5, T=10, fs=500, transform=None, save=False, varyFreq=False,
                 target='freq'):
        """
        Initialize a experimental sinusoid signal dataset to test transforms with.

        :param size: number of independent signals contained within the dataset
        :param sinusoids: number of sinusoid functions summed in each signal.
        :param max_freq: if load is False, generate a dataset with intrinsic frequencies capped by max_freq
        :param T: time of signal in seconds
        :param fs: sample frequency
        :param transform: pytorch transform object to apply to the signal in __getitem__
        :param save: bool whether to save the generated dataset to the disk
        :param varyFreq: bool whether to change the structure of the generating sin function to allow for varying
                         frequency in time.  i.e. sin(2pi * f(t) * t) where f(t) is linear by a random slope

        main instance variables:
        self.data : shape=(size, self.T * self.fs), contains each signal as a row
        self.data_params_freq : shape=(size, sinusoids), contains the frequency parameters that generated each signal, by row
        self.data_params_amp : shape=(size, sinusoids), contains the amplitude parameters that generated each signal, by row
        """

        self.transform = transform
        self.n = size
        # Time of signal in seconds
        self.T = T
        # How are these seconds sampled, Hz?
        self.fs = fs
        self.sinusoids = sinusoids
        self.vary_freq = varyFreq
        # Generate new dataset
        self.max_freq = max_freq
        self.data = torch.zeros(size, self.T * self.fs)
        self.data_params_freq = torch.zeros(size, sinusoids)
        self.data_params_amp = torch.zeros(size, sinusoids)
        self.data_params_timevaryingfreq = torch.zeros(size, sinusoids)
        if target == "freq":
            self.target = self.data_params_freq
        elif target == "freq_vary":
            self.target = self.data_params_timevaryingfreq
        elif target == "amp":
            self.target = self.data_params_amp

        # Generate the signals
        for i in range(size):
            if varyFreq:
                self.data[i], self.data_params_timevaryingfreq[i], self.data_params_amp[i] = self.generateVaryingTimeSignal()
            else:
                self.data[i], self.data_params_freq[i], self.data_params_amp[i] = self.generateSignal()
        if save:
            torch.save(self.data, "data-sins-" + str(sinusoids) + "-len-" + str(size) + ".pt")
            torch.save(self.data_params_amp, "amp-sins-" + str(sinusoids) + "-len-" + str(size) + ".pt")
            torch.save(self.data_params_freq, "freq-sins-" + str(sinusoids) + "-len-" + str(size) + ".pt")



    def __len__(self):
        return self.n


    def __getitem__(self, idx):
        # Target here is frequency of the sin wave
        if self.transform is not None:
            return self.transform(self.data[idx]), self.target[idx]
        return self.data[idx], self.target[idx]


    def viewTrueSignal(self, idx):
        """
        Generate a 2D plot of the non-transformed signal using web browser and plotly
        Add text revealing the true frequency and amplitude coefficients of the signal
        :param item: the index of signal to view
        """
        fig = go.Figure(go.Scatter(y=self.data[idx], x=np.linspace(0, self.T, self.T*self.fs)))
        if self.vary_freq:
            fig.update_layout(title="f(x) = sin(2*pi * " + str(self.data_params_timevaryingfreq[idx,0]) + " x * x)")
        else:
            fig.update_layout(title="freq:" + str(self.data_params_freq[idx])
                                    + "\n\namp:" + str(self.data_params_amp[idx]))
        fig.show()


    def viewSignal(self, idx):
        """
        Generate a 2D plot of the (possibly TRANSFORMED) signal using web browser and plotly
        Add text revealing the true frequency and amplitude coefficients of the signal
        :param idx: the index of signal to view
        """
        if self.transform is None: return self.viewTrueSignal(idx)

        if len(self.transform.domain) == 1:
            # One dimensional
            fig = go.Figure(go.Scatter(x=self.transform.domain[0], y=self.transform(self.data[idx])))
        elif len(self.transform.domain) == 2:
            # Transform is an image
            fig = go.Figure(data=go.Heatmap(z=self.transform(self.data[idx]),
                            x=self.transform.domain[1],
                            y=self.transform.domain[0]))
            fig.update_yaxes(title_text="Wavelet scale", type='category')
            fig.update_xaxes(title_text="Time (seconds)", type='category')
        else: raise ValueError("Shape of the domain is not handled for visualizing")

        fig.update_layout(title="freq:" + str(self.data_params_freq[idx])
                                + "\n\namp:" + str(self.data_params_amp[idx]))
        fig.show()


    def generateSignal(self):
        """
        :return: the sin signal generated by random frequency and amplitude parameters
        """
        signal = torch.zeros(self.T * self.fs)
        # Obtain random frequency and amplitude coefficients for each sinusoid
        freq_coefs = np.random.uniform(0.1, self.max_freq, self.sinusoids)
        amp_coefs = np.random.uniform(1.5, 5.5, self.sinusoids)
        # Obtain the signal by evaluating the sinusoids
        for i,v in enumerate((np.linspace(0, self.T, self.T * self.fs))):
            signal[i] = sum(amp_coefs[j] * np.sin(2*np.pi * freq_coefs[j] * v) for j in range(self.sinusoids))
        return signal, torch.tensor(freq_coefs), torch.tensor(amp_coefs)


    def generateVaryingTimeSignal(self):
        """
        :return: the sin signal generated by random frequency and amplitude parameters
        """
        signal = torch.zeros(self.T * self.fs)
        # Obtain random frequency and amplitude coefficients for each sinusoid
        amp_coefs = np.random.uniform(1.5, 5.5, self.sinusoids)
        time_varying_coefs = np.random.uniform(0.25, self.max_freq, self.sinusoids)
        # Obtain the signal by evaluating the sinusoids
        for i,v in enumerate((np.linspace(0, self.T, self.T * self.fs))):
            signal[i] = sum(amp_coefs[j] * np.sin(2*np.pi * time_varying_coefs[j] * v * v) for j in range(self.sinusoids))
        return signal, torch.tensor(time_varying_coefs), torch.tensor(amp_coefs)