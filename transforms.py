"""
transforms.py

Classes implementing common digital signal transformations:
Fourier
Short Time Fourier
PowerSpectrum
Wavelet

All transforms default signal time and sampling rate are set to T=10, fs=500 respectively, but remain modifiable
All transforms contain a __call__(signal) which is to be used in a pytorch dataset
    Signal is assumed to be of shape=(n,) of real valued floats
All transforms utilize numpy and not torch tensors
All transforms contain a domain class variable and a domain_shape, outlining the tensor size of the resulting transform
    domain (dict), key 0: values of first axis, key 1: values of second axis (if 2D)
All transforms have a .view(signal) member function to visualize the transformed signal in plotly.graph_objects

@author Rylan Marianchuk
September 2021
"""
import plotly.graph_objs as go
import numpy as np
import pywt
import math


class Fourier:
    def __init__(self, T=10, fs=500):
        """
        :param T: length of signal in seconds
        :param fs: (int) sampling rate of signal
        """
        self.fs = fs
        self.T = T
        self.domain = {0: np.fft.rfftfreq(T*fs, 1/fs)}
        self.domain_shape = len(self.domain[0])
        return

    def __call__(self, signal):
        """
        Invoke numpy's real valued Fourier Transform
        :param signal: (float) shape=(n,) to transform
        :return: (complex) array of shape=domain_shape
        """
        return np.fft.rfft(signal)

    def magnitude(self, signal):
        """
        Obtain the magnitude of the returned complex values of Fourier
        :param signal: (float) shape=(n,) to transform
        :return: (float64) shape=domain_shape magnitude of all complex components
        """
        complex = self(signal)
        return np.abs(complex)

    def phase(self, signal):
        """
        Obtain the phase (angle) of the returned complex values of Fourier
        :param signal: (float) shape=(n,) to transform
        :return: (float64) shape=domain_shape phase of all complex components
        """
        complex = self(signal)
        return np.angle(complex)

    def viewComplex(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable 2D complex plane of the Fourier transform
        """
        trfm = self(signal)
        fig = go.Figure(go.Scatter(x=np.real(trfm), y=np.imag(trfm), mode='markers'))
        fig.update_layout(title="Complex Plane of Fourier Transform")
        fig.update_xaxes(title_text="Real")
        fig.update_yaxes(title_text="Imaginary")
        fig.show()

    def viewMagnitude(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable violin plot of the magnitude of all complex components
        """
        fig = go.Figure(go.Violin(y=self.magnitude(signal), box_visible=True, points="all", name="Distribution Shape"))
        fig.update_layout(title="Fourier Magnitude Distribution")
        fig.update_yaxes(title_text="Magnitude")
        fig.show()

    def viewPhase(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable violin plot of the phase of all complex components
        """
        fig = go.Figure(go.Violin(y=self.phase(signal), box_visible=True, points="all", name="Distribution Shape"))
        fig.update_layout(title="Fourier Phase Distribution")
        fig.update_yaxes(title_text="Phase")
        fig.show()


class stFourier:
    def __init__(self, window_size, T=10, fs=500, log=False, normalize=False):
        """
        :param window_size: (int) number of signal elements to use in transform
        :param T: length of signal in seconds
        :param fs: (int) sampling rate of signal
        :param log: (bool) whether to take the logarithm of the resulting power
        :param normalize: (bool) whether to scale all results between 0 and 1, preserving the distances
        """
        assert (window_size < T*fs), "Specified window size greater than the signal length itself"
        self.fs = fs
        self.T = T
        self.log = log
        self.normalize = normalize
        self.win = window_size
        self.overlap_frac = 0.5
        self.jump = math.floor(window_size * self.overlap_frac)
        self.n_windows = int(math.ceil(self.T*self.fs / self.jump))
        self.domain = {0: np.array(range(0, T*fs, self.jump)) / fs,
                       1: np.fft.rfftfreq(self.win, 1/fs)}
        self.domain_shape = (len(self.domain[0]), len(self.domain[1]))
        self.hanning = np.hanning(self.win)

    def __call__(self, signal):
        """
        Slide window across signal taking the Fourier Transform at each
        :param signal: (float) shape=(n,) to transform
        :return: (float) shape=domain_shape 2D image of the stFT, each row a Power Spectrum at a given window start time
        """
        assert (signal.shape[0] == self.T*self.fs), "The signal is not corresponding to the specified time length and" \
                                                    "sample frequency"
        result = np.zeros(shape=(self.domain_shape[0], self.domain_shape[1]))
        signal_pad = np.concatenate((signal, np.zeros(self.win)))
        for i, L_edge in enumerate((range(0, signal.shape[0], self.jump))):
            dampened = self.hanning * signal_pad[L_edge: L_edge + self.win]
            F = np.fft.rfft(dampened)
            result[i] = np.power(np.abs(F), 2)
            if self.log: result[i] = np.log(result[i])

        if self.normalize:
            return result / np.linalg.norm(result)
        return result

    def view(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable Heatmap plot of the power at a given window and frequency
        """
        trfm = self(signal)
        fig = go.Figure(data=go.Heatmap(z=trfm, x=self.domain[1], y=self.domain[0]))
        fig.update_layout(title="Short Time Fourier Transform  -  Window size: " + str(self.win))
        fig.update_yaxes(title_text="Window start time (seconds)", type='category')
        fig.update_xaxes(title_text="Power Spectrum of Window (frequency)", type='category')
        fig.show()


class PowerSpec:
    def __init__(self, T=10, fs=500, log=False, normalize=False):
        """
        :param T: length of signal in seconds
        :param fs: (int) sampling rate of signal
        :param log: (bool) whether to take the logarithm of the resulting power
        :param normalize: (bool) whether to scale all results between 0 and 1, preserving the distances
        """
        self.fs = fs
        self.T = T
        self.log = log
        self.normalize = normalize
        self.FT = Fourier(T=T, fs=fs)
        self.domain = {0: np.fft.rfftfreq(T*fs, 1/fs)}
        self.domain_shape = len(self.domain[0])
        return

    def __call__(self, signal):
        """
        Obtain the power spectrum of the signal
        :param signal: (float) shape=(n,) to transform
        :return: (float) shape=domain_shape 1D Power Spectrum (magnitude of the FT squared)
        """
        if self.log:
            return np.log(np.power(self.FT.magnitude(signal), 2))
        powerSpec = np.power(self.FT.magnitude(signal), 2)
        if self.normalize:
            return powerSpec / np.linalg.norm(powerSpec)
        return powerSpec

    def view(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable scatter plot of the Power of all frequencies in signal
        """
        trfm = self(signal)
        fig = go.Figure(go.Scatter(x=self.domain[0], y=trfm))
        fig.update_layout(title="PowerSpectrum Transform")
        fig.update_xaxes(title_text="Frequency")
        fig.update_yaxes(title_text="Power  -  log=" + str(self.log))
        fig.show()


class Wavelet:
    def __init__(self, widths, wavelet='mexh', T=10, fs=500, normalize=False):
        """
        :param widths: (1D array) sequence of increasing wavelet widths, used as scale axis
        :param wavelet: (str) the chosen wavelet type. Options may be viewed with .seeAvailableWavelets()
        :param T: length of signal in seconds
        :param fs: (int) sampling rate of signal
        :param normalize: (bool) whether to scale all results between 0 and 1, preserving the distances
        """
        self.fs = fs
        self.T = T
        self.normalize = normalize
        self.wavelet = wavelet
        self.widths = widths
        self.domain = { 0 : widths, # each row a wavelet width is selected and translated across time
                        1: np.linspace(0, T, T*fs)} # linear time domain sequence
        self.domain_shape = (len(widths), T*fs)
        return

    def __call__(self, signal):
        """
        Transform the signal to its image of wavelet coefficients
        :param signal: (float) shape=(n,) to transform
        :return: (float) shape=domain_shape 2D image of wavelet coefs
        """
        if self.normalize:
            trfm = pywt.cwt(signal, self.widths, self.wavelet)[0]
            return trfm / np.linalg.norm(trfm)
        return pywt.cwt(signal, self.widths, self.wavelet)[0]

    def seeAvailableWavelets(self):
        print(pywt.wavelist(kind='continuous'))

    def view(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable Heatmap plot viewing all wavelet coefficients at a given scale and translation time
        """
        trfm = self(signal)
        fig = go.Figure(data=go.Heatmap(z=trfm, x=self.domain[1], y=self.domain[0]))
        fig.update_layout(title="Wavelet Transform  -  Wavelet: " + str(self.wavelet))
        fig.update_yaxes(title_text="Wavelet scale", type='category')
        fig.update_xaxes(title_text="Time (seconds)", type='category')
        fig.show()


class GramianAngularField:
    def __init__(self, T=10, fs=500):
        self.T = T
        self.fs = fs

    def __call__(self):
        return


class MarkovTransitionField:
    def __init__(self, T=10, fs=500):
        self.T = T
        self.fs = fs

    def __call__(self):
        return


# ---------------- Inverse Transforms ----------------

class invFourier(object):
    def __init__(self, T=10, fs=500):
        self.fs = fs
        self.T = T
        self.domain = {0: np.linspace(0, T, T*fs)}
        self.domain_shape = T*fs
        return

    def __call__(self, signal):
        return np.fft.irfft(signal)
