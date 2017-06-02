import numpy as np
import matplotlib.pyplot as plt


x = np.random.random(20)

print(x)
print('\n')

spectrum = np.fft.fft(np.sin(x))
freq = np.fft.fftfreq(x.shape[-1])

print(abs(spectrum))

plt.plot(freq, abs(spectrum))
plt.show()