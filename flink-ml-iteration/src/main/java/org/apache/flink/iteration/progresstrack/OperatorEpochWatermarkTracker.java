/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.iteration.progresstrack;

import org.apache.flink.annotation.VisibleForTesting;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.flink.util.Preconditions.checkNotNull;
import static org.apache.flink.util.Preconditions.checkState;

/**
 * Tracks the epoch watermark from each input. Once the minimum epoch watermark changed, it would
 * notify the listener.
 */
public class OperatorEpochWatermarkTracker {

    private final OperatorEpochWatermarkTrackerListener progressTrackerListener;

    private final List<InputStatus> inputStatuses;

    private final LowerBoundMaintainer allInputsLowerBound;

    public OperatorEpochWatermarkTracker(
            int[] numberOfChannels, OperatorEpochWatermarkTrackerListener progressTrackerListener) {
        checkState(numberOfChannels != null && numberOfChannels.length >= 1);
        this.progressTrackerListener = checkNotNull(progressTrackerListener);

        this.inputStatuses = new ArrayList<>(numberOfChannels.length);
        for (int numberOfChannel : numberOfChannels) {
            inputStatuses.add(new InputStatus(numberOfChannel));
        }

        this.allInputsLowerBound = new LowerBoundMaintainer(numberOfChannels.length);
    }

    public void onEpochWatermark(int inputIndex, String sender, int epochWatermark)
            throws IOException {
        InputStatus inputStatus = inputStatuses.get(inputIndex);
        inputStatus.onUpdate(sender, epochWatermark);

        if (inputStatus.getInputLowerBound() > allInputsLowerBound.getValue(inputIndex)) {
            int oldLowerBound = allInputsLowerBound.getLowerBound();
            allInputsLowerBound.updateValue(inputIndex, inputStatus.getInputLowerBound());
            if (allInputsLowerBound.getLowerBound() > oldLowerBound) {
                progressTrackerListener.onEpochWatermarkIncrement(
                        allInputsLowerBound.getLowerBound());
            }
        }
    }

    @VisibleForTesting
    int[] getNumberOfInputs() {
        return inputStatuses.stream()
                .mapToInt(inputStatus -> inputStatus.numberOfChannels)
                .toArray();
    }

    private static class InputStatus {
        private final int numberOfChannels;
        private final Map<String, Integer> senderIndices;
        private final LowerBoundMaintainer allChannelsLowerBound;

        public InputStatus(int numberOfChannels) {
            this.numberOfChannels = numberOfChannels;
            this.senderIndices = new HashMap<>(numberOfChannels);
            this.allChannelsLowerBound = new LowerBoundMaintainer(numberOfChannels);
        }

        public void onUpdate(String sender, int epochWatermark) {
            int index = senderIndices.computeIfAbsent(sender, k -> senderIndices.size());
            checkState(index < numberOfChannels);

            allChannelsLowerBound.updateValue(index, epochWatermark);
        }

        public int getInputLowerBound() {
            return allChannelsLowerBound.getLowerBound();
        }
    }

    private static class LowerBoundMaintainer {

        private final int[] values;

        private int lowerBound;

        public LowerBoundMaintainer(int numberOfValues) {
            this.values = new int[numberOfValues];
            Arrays.fill(values, Integer.MIN_VALUE);
            lowerBound = Integer.MIN_VALUE;
        }

        public int getLowerBound() {
            return lowerBound;
        }

        public int getValue(int channel) {
            return values[channel];
        }

        public void updateValue(int channel, int value) {
            checkState(
                    value > values[channel],
                    String.format(
                            "The channel %d received an outdated value %d, which currently is %d",
                            channel, value, values[channel]));
            if (value > values[channel]) {
                long oldValue = values[channel];
                values[channel] = value;

                if (oldValue == lowerBound) {
                    lowerBound = calculateLowerBound();
                }
            }
        }

        private int calculateLowerBound() {
            int newLowerBound = values[0];
            for (int i = 1; i < values.length; ++i) {
                if (values[i] < newLowerBound) {
                    newLowerBound = values[i];
                }
            }

            return newLowerBound;
        }
    }
}
