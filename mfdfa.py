import numpy as np

def dfa(x, S, m, Q, skip_agg=False):
    """
        Calculate generalized Hurst Expornenet by Multi Fractal Detrended Fluctuation Analysis.

        Mainly reference paper `Multifractal detrended fluctuation analysis: Practical
        applications to financial time series`
        http://www.ise.ncsu.edu/jwilson/files/mfdfa-pafts.pdf

        And some other references.
        `Multifractal Detrended Fluctuation Analysis of Nonstationary Time Series`
        https://arxiv.org/pdf/physics/0202070.pdf
        `Introduction to Multifractal Detrended Fluctuation Analysis in Matlab`
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3366552/

        Args:
            x(array(float))  : Target time series data.
            S(array(int))    : Intervals which divides culmative sum time series. Positive Integer
                               value satisfy condition `20 ≤ s ≤ N/10` is recommended
                               by http://www.ise.ncsu.edu/jwilson/files/mfdfa-pafts.pdf
            m(int)           : degree of polynomial fit for each segment.
            Q(array(int))    : Array of fluctuation q-th order.
            skip_agg(bool)   : Whether use cumsum for profile, for finantial time series or randam walk,
                               Its not needed.
        Returns:
            array(float)     : Return generalized hurst expornent array of each Q-th order.
    """
    N = len(x)
    if skip_agg:
        y = x
    else:
        y = np.cumsum(x - np.mean(x))

    def Fvs2(v, s, reverse=False):
        """
            (Root Mean Square) ** 2
            F(s, v)
        """
        Ns = int(N//s)
        ax = np.arange(1, s+1)
        if reverse:
            segment = y[N - (v-Ns)*s:N - (v-Ns)*s+s]
        else:
            segment = y[(v-1)*s:v*s]
        coef = np.polyfit(ax, segment, m)
        fitting = np.polyval(coef, ax)
        return np.mean((segment - fitting)**2)

    Fhq = np.zeros(len(Q))
    for i, q in enumerate(Q):
        Fqs = np.zeros(len(S))

        for j, s in enumerate(S):
            Ns = int(N//s)
            segs = np.array([
                [Fvs2(v, s) for v in range(1, Ns + 1)],
                [Fvs2(v, s, reverse=True) for v in range(Ns+1, 2 * Ns + 1)]
            ]).reshape(-1)

            assert len(segs) == 2 * Ns, '{} segments'.format(len(segs))

            if q == 0:
                Fqs[j] = np.exp(np.mean(np.log(segs))/2)
            else:
                Fqs[j] = np.mean(segs ** (q/2)) ** (1/q)

        coef = np.polyfit(np.log(S), np.log(Fqs), 1)
        Fhq[i] = coef[0]

    return Fhq


def basic_dfa(x, Q, skip_agg=False, observations=100):
    """
        Use supposed appropriate parameters in
        http://www.ise.ncsu.edu/jwilson/files/mfdfa-pafts.pdf
    """
    N = len(x)
    s_min = max(20, int(np.floor(N/100)))
    s_max = min(20*s_min, int(np.floor(N/10)))
    s_inc = (s_max - s_min) / observations
    S = [s_min + int(np.floor(i*s_inc)) for i in range(0, observations)]
    return dfa(x, S=S, m=1, Q=Q, skip_agg=skip_agg)


def hurst(x, skip_agg=False, observations=100):
    """
        Calculate Hurst Expoenent.
    """
    return basic_dfa(x, Q=[2],
                        skip_agg=skip_agg,
                        observations=observations)[0]
