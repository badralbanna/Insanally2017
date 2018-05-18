from .single_ISI_in_time import timed_prob, set_bw_ISI_in_time, BW_ISI, WINDOW, STEP
from .PSTH import PSTH_poisson_prob_2, set_bw_PSTH, BW_PSTH, BW_METHOD


################################################
# Joint Probability of the ISI and Firing rate #
################################################

# use with lock = False, log = True, prob_from_spikes = False, latency = True
def joint_prob(values, times, **kwargs):
    ISI_prob = timed_prob(values, times, **kwargs)
    PSTH_prob = PSTH_poisson_prob_2(times, **kwargs)

    def joint_prob_func(ISI, time):
        p_ISI = ISI_prob(ISI, time)
        p_PSTH = PSTH_prob(ISI, time)
        prob = p_ISI * p_PSTH
        # print('p_ISI: {0}, p_PSTH: {1}, prob: {2}'.format(p_ISI, p_PSTH, prob))
        return(prob)
    
    return(joint_prob_func)

def set_bw_PSTH_and_bw_ISI(ISIs, times, **kwargs):
    bw = set_bw_ISI_in_time(ISIs, times, **kwargs)
    bw.update(set_bw_PSTH(times, **kwargs))
    return(bw) 
