import numpy as np
from lifelines import AalenJohansenFitter, KaplanMeierFitter

def correction(times, events, groups, horizon, predictions = None, cause=1):
    """Group-level corrective factor c_r(horizon, g) = F^C_r / F^NC_r.

    times   : observed times  T = min(T', C)
    events  : 0 = censored, `cause` = event of interest, other ints = competing
    groups  : group label per individual
    horizon : tau, the decision horizon
    """
    out = {}
    for g in np.unique(groups):
        m = groups == g
        t, e = times[m], events[m]

        # Numerator: F^C_r(tau|g) via Aalen-Johansen CIF for `cause`
        ajf = AalenJohansenFitter(calculate_variance=False, seed = 42)
        ajf.fit(t, e, event_of_interest=cause)
        f_c = ajf.predict(horizon)

        # Denominator: F^NC_r(tau|g) = 1 - cause-r KM (competing events censored)
        if predictions is None:
            kmf = KaplanMeierFitter()
            kmf.fit(t, event_observed=(e == cause))
            f_nc = 1 - kmf.predict(horizon)
        else:
            f_nc = predictions[m].mean()

        out[g] = f_c / f_nc if f_nc > 0 else np.nan
    return out