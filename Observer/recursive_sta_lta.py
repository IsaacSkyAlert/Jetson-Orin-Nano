def recursive_sta_lta_py(a, nsta, nlta):
    """
    Recursive STA/LTA written in Python.

    .. note::

        There exists a faster version of this trigger wrapped in C
        called :func:`~obspy.signal.trigger.recursive_sta_lta` in this module!

    :type a: NumPy :class:`~numpy.ndarray`
    :param a: Seismic Trace
    :type nsta: int
    :param nsta: Length of short time average window in samples
    :type nlta: int
    :param nlta: Length of long time average window in samples
    :rtype: NumPy :class:`~numpy.ndarray`
    :return: Characteristic function of recursive STA/LTA

    .. seealso:: [Withers1998]_ (p. 98) and [Trnkoczy2012]_
    """
    ndat = len(a)
    # compute the short time average (STA) and long time average (LTA)
    # given by Evans and Allen
    csta = 1. / nsta
    clta = 1. / nlta
    sta = 0.
    lta = np.finfo(0.0).tiny  # avoid zero division
    a = np.square(a)
    charfct = np.zeros(ndat, dtype=np.float64)
    icsta = 1 - csta
    iclta = 1 - clta
    for i in range(1, ndat):
        sta = csta * a[i] + icsta * sta
        lta = clta * a[i] + iclta * lta
        charfct[i] = sta / lta
    charfct[:nlta] = 0

    return charfct
