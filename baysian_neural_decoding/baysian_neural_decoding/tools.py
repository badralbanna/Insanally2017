import numpy


def flatten(responses):
    return(numpy.array([i for j in responses for i in j]))


def filter_times(times, offsets, ranges, responses=None):
    new_times = []
    if responses is not None:
        new_responses = []
    for i, (time, offset, (min_time, max_time)) in enumerate(zip(times, offsets, ranges)):
        new_time = numpy.array(time - offset)
        index = (new_time > min_time) * (new_time < max_time)
        new_time = new_time[index]
        new_times.append(new_time.copy())
        if responses is not None:
            new_response = responses[i][index]
            new_responses.append(new_response.copy())
    if responses is not None:
        return(numpy.array(new_times), numpy.array(new_responses))
    else:
        return(numpy.array(new_times), None)
