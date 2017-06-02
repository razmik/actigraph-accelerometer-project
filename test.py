import numpy as np
import matplotlib.pyplot as plt


# x = np.random.random(100)
x = np.array([1,2,1,0,1,2,1,0])

spectrum = np.fft.fft(np.sin(x))
freqs = np.fft.fftfreq(x.shape[-1], 0.01)

idx = np.argmax(np.abs(spectrum))
freq = freqs[idx]

print("res", freq, np.amax(np.abs(spectrum)))

plt.plot(freqs, abs(spectrum))
plt.show()
