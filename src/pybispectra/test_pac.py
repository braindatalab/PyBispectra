import numpy as np
from pybispectra.cfc import PAC, PPC
from pybispectra.utils import compute_fft

if __name__ == "__main__":
    rand = np.random.RandomState(44)
    data = rand.rand(10, 3, 200)
    sfreq = 100
    n_jobs = 1

    fft, freqs = compute_fft(data, sfreq, n_jobs)

    ppc = PPC(fft, freqs)
    ppc.compute(n_jobs=n_jobs)
    ppc_results = ppc.results

    # pac = PAC(fft, freqs)
    # pac.compute(symmetrise="none", normalise="none", n_jobs=n_jobs)
    # pac_results = pac.results
    # pac_results[0].plot()

    print("jeff")
