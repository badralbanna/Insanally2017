import numpy


#####################
# Finding Latencies #
#####################

def first_spike_latency(response, log=False, **kwargs):
    if len(response) > 0:
        time = [response[response > 0][0]]
    else:
        time = []

    if not log:
        return(numpy.array(time), numpy.array(time))
    else:
        return(numpy.log10(numpy.array(time)), numpy.array(time))

############################
# Estimating Probabilities #
############################

# Use set_bw and KDE wrapper
