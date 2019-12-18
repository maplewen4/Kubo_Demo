import numpy as np
import scipy.fftpack as fourier_transform


class Kubo(object):
    """
    The kubo model contains all attributions: C(t), g(t), I(w)
    
    Attributes
    ----------
        tau : float
            The decay constant. Must be the same unit with time t. AUt is preferred.
        delta : float
            The fluctuation. AUE is preferred.

    Methods
    -------
        calculate_Ct : calculate time correlation function.
        calculate_gt : calculate lineshape function.
        calculate_Iw : calculate lineshape.    
    
    """
    def __init__(self, tau, delta):

        self.tau = tau
        self.delta = delta
        self.unit_conv = {"cm1_to_phz": 2.997925e-5, "hz_to_cm1": 3.335641e-11}

    def calculate_Ct(self, time):
        """contains method to generate C(t), the TCF, from tau and delta"""

        # define the method to calculate C(t)
        ct_list = self.delta**2 * np.exp(-time / self.tau)

        t_vs_Ct = np.column_stack((time, ct_list))

        return t_vs_Ct

    def calculate_gt(self, time):
        """contains method to generate g(t) from C(t)"""

        # unit conv
        unit_conv_gt = self.unit_conv["cm1_to_phz"]

        # define the method to calculate g(t)
        gt_list_unit = self.delta**2 * self.tau * time + self.delta**2 * self.tau**2 * (np.exp(-time / self.tau) - 1)
        gt_list = unit_conv_gt**2 * gt_list_unit

        t_vs_gt = np.column_stack((time, gt_list))

        return t_vs_gt

    def calculate_Iw(self, time):
        """contains method to generate I(w), the final spectrum, from g(t)"""

        # manipulate time and gt
        # tlist
        t_list_half = time.copy()

        t_list_extra_tail = np.append(t_list_half, (t_list_half + np.amax(t_list_half)))
        t_list = t_list_extra_tail[:-1]

        # egt list
        egt_list_half = np.exp(-(self.gt(time)[:, 1]))

        egt_list_half_conjugate = np.conj(egt_list_half)
        egt_list_half_mirror = np.flipud(egt_list_half_conjugate)[:-1]

        egt_list = np.append(egt_list_half, egt_list_half_mirror)

        # do fft
        iw_list = np.real(fourier_transform.fft(egt_list))

        # generate w list
        # unit conv
        unit_conv_iw = self.unit_conv["hz_to_cm1"]

        # fftfreq the list
        w_size = len(t_list)
        dt = t_list[1] - t_list[0]

        w_list_unitless = fourier_transform.fftfreq(w_size, d=dt * 1e-15)
        w_list = w_list_unitless * unit_conv_iw

        w_vs_Iw = np.column_stack((w_list, iw_list))

        return w_vs_Iw
