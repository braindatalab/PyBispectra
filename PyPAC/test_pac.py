import numpy as np
from init import PPC, compute_fft

if __name__ == "__main__":
    np.random.RandomState(44)
    data = np.random.rand(30, 10, 1000)
    sfreq = 500

    fft, freqs = compute_fft(data, sfreq)

    ppc = PPC(fft, freqs)
    ppc.compute(n_jobs=3)
    raveled_results = ppc.get_results("raveled")
    compact_results, compact_indices = ppc.get_results("compact")

    print("jeff")
