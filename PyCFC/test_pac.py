import numpy as np
from init import PAC, PPC, compute_fft

if __name__ == "__main__":
    np.random.RandomState(44)
    data = np.random.rand(10, 3, 200)
    sfreq = 100
    n_jobs = 9

    fft, freqs = compute_fft(data, sfreq, n_jobs)

    ppc = PPC(fft, freqs)
    ppc.compute(n_jobs=n_jobs)
    raveled_results = ppc.get_results("raveled")
    compact_results, compact_indices = ppc.get_results("compact")

    pac = PAC(fft, freqs)
    pac.compute(n_jobs=n_jobs)
    raveled_pac, raveled_pac_types = pac.get_results("raveled")
    compact_pac, compact_pac_types, pac_indices = pac.get_results("compact")

    print("jeff")
