################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from abc import ABC

from pyflink.common.time import Time


class Windows(ABC):
    """
    Windowing strategy that determines how to create mini-batches from input data.
    """
    pass


class GlobalWindows(Windows):
    """
    A windowing strategy that groups all elements into a single global window.
    This strategy assumes that the input strategy is bounded.
    """

    def __eq__(self, other):
        return isinstance(other, GlobalWindows)


class CountTumblingWindows(Windows):
    """
    A windowing strategy that groups elements into fixed-size windows based on
    the count number of the elements. Windows do not overlap.
    """

    def __init__(self, size: int):
        super().__init__()
        self._size = size

    @staticmethod
    def of(size: int) -> 'CountTumblingWindows':
        return CountTumblingWindows(size)

    @property
    def size(self) -> int:
        return self._size

    def __eq__(self, other):
        return isinstance(other, CountTumblingWindows) and self._size == other._size


class EventTimeTumblingWindows(Windows):
    """
    A windowing strategy that groups elements into fixed-size windows based on
    the timestamp of the elements. Windows do not overlap.
    """

    def __init__(self, size: Time):
        super().__init__()
        self._size = size

    @staticmethod
    def of(size: Time) -> 'EventTimeTumblingWindows':
        return EventTimeTumblingWindows(size)

    @property
    def size(self) -> Time:
        return self._size

    def __eq__(self, other):
        return isinstance(other, EventTimeTumblingWindows) and self._size == other._size


class ProcessingTimeTumblingWindows(Windows):
    """
    A windowing strategy that groups elements into fixed-size windows based on
    the current system time of the machine the operation is running on. Windows
    do not overlap.
    """

    def __init__(self, size: Time):
        super().__init__()
        self._size = size

    @staticmethod
    def of(size: Time) -> 'ProcessingTimeTumblingWindows':
        return ProcessingTimeTumblingWindows(size)

    @property
    def size(self) -> Time:
        return self._size

    def __eq__(self, other):
        return isinstance(other, ProcessingTimeTumblingWindows) and self._size == other._size


class EventTimeSessionWindows(Windows):
    """
    A windowing strategy that groups elements into sessions based on the
    timestamp of the elements. Windows do not overlap.
    """

    def __init__(self, gap: Time):
        super().__init__()
        self._gap = gap

    @staticmethod
    def with_gap(gap: Time) -> 'EventTimeSessionWindows':
        return EventTimeSessionWindows(gap)

    @property
    def gap(self) -> Time:
        return self._gap

    def __eq__(self, other):
        return isinstance(other, EventTimeSessionWindows) and self._gap == other._gap


class ProcessingTimeSessionWindows(Windows):
    """
    A windowing strategy that groups elements into sessions based on the current
    system time of the machine the operation is running on. Windows do
    not overlap.
    """

    def __init__(self, gap: Time):
        super().__init__()
        self._gap = gap

    @staticmethod
    def with_gap(gap: Time) -> 'ProcessingTimeSessionWindows':
        return ProcessingTimeSessionWindows(gap)

    @property
    def gap(self) -> Time:
        return self._gap

    def __eq__(self, other):
        return isinstance(other, ProcessingTimeSessionWindows) and self._gap == other._gap
