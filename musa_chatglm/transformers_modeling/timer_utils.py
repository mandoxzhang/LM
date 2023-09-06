# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import torch
from numpy import mean, percentile

try:
    from megatron import print_rank_0
except:
    print_rank_0 = print


# try:
#     import psutil

#     PSUTILS_INSTALLED = True
# except ImportError:
#     PSUTILS_INSTALLED = False
#     pass
try:
    import torch_musa

    current_stream = torch.musa.current_stream
    event_class = torch.musa.Event
    print("using musa event")
except:
    current_stream = torch.cuda.current_stream
    event_class = torch.cuda.Event
    print("using cuda event")


class EventTimer(object):
    def __init__(self, start_event, end_event):
        self.start_event = start_event
        self.end_event = end_event

    def get_elapsed_msec(self):
        s0 = current_stream()
        s0.wait_event(self.end_event)
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""

    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.started_ = False
            self.event_timers = []
            self.use_host_timer = False
            self.start_event = None
            self.elapsed_records = None
            self.start_time = 0.0
            self.end_time = 0.0

        def start(self):
            """Start the timer."""
            assert not self.started_, f"{self.name_} timer has already been started"
            if self.use_host_timer:
                self.start_time = time.perf_counter()
            else:
                # event_class = torch.musa.Event
                self.start_event = event_class(enable_timing=True)
                self.start_event.record()
            self.started_ = True

        def stop(self, reset=False, record=False):
            """Stop the timer."""
            assert self.started_, "timer is not started"
            # event_class = torch.musa.Event
            if self.use_host_timer:
                self.end_time = time.perf_counter()
                self.event_timers.append(self.end_time - self.start_time)
            else:
                # event_class = torch.musa.Event
                end_event = event_class(enable_timing=True)
                end_event.record()
                self.event_timers.append(EventTimer(self.start_event, end_event))
                self.start_event = None
            self.started_ = False

        def _get_elapsed_msec(self):
            if self.use_host_timer:
                self.elapsed_records = [et * 1000.0 for et in self.event_timers]
            else:
                self.elapsed_records = [
                    et.get_elapsed_msec() for et in self.event_timers
                ]
            self.event_timers.clear()
            return sum(self.elapsed_records)

        def reset(self):
            """Reset timer."""
            self.started_ = False
            self.start_event = None
            self.elapsed_records = None
            self.event_timers.clear()

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self._get_elapsed_msec()
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

        def mean(self):
            self.elapsed(reset=False)
            return trim_mean(self.elapsed_records, 0.0)

    def __init__(self):
        self.timers = {}

    def get_timers(self):
        return self.timers

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def get_mean(self, names, normalizer=1.0, reset=True):
        """Get the mean of a group of timers."""
        assert normalizer > 0.0
        means = {}
        for name in names:
            if name in self.timers:
                elapsed_time = self.timers[name].mean() * 1000.0 / normalizer
                means[name] = elapsed_time
        return means

def trim_mean(data, trim_percent):
    """Compute the trimmed mean of a list of numbers.

    Args:
        data (list): List of numbers.
        trim_percent (float): Percentage of data to trim.

    Returns:
        float: Trimmed mean.
    """
    assert trim_percent >= 0.0 and trim_percent <= 1.0
    n = len(data)
    # Account for edge case of empty list
    if len(data) == 0:
        return 0
    data.sort()
    k = int(round(n * (trim_percent)))
    percent = [0, 25, 50, 75, 90, 95, 100]
    string = "*" * 15 + "percentile" + "*" * 15
    for p in percent:
        string += f" {p}:{percentile(data, p)} "
    print_rank_0(string)
    return mean(data[k : n - k])
