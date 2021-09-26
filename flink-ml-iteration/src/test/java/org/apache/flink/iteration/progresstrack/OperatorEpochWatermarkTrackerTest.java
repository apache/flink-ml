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

import org.apache.flink.util.TestLogger;

import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

/** Tests the logic of {@link OperatorEpochWatermarkTracker}. */
public class OperatorEpochWatermarkTrackerTest extends TestLogger {

    @Test
    public void testEpochWatermarkAlignment() throws IOException {
        RecordingProgressListener recordingProgressListener = new RecordingProgressListener();
        int[] numberOfChannels = new int[] {2, 3};
        OperatorEpochWatermarkTracker progressTracker =
                new OperatorEpochWatermarkTracker(numberOfChannels, recordingProgressListener);

        testOnEpochWatermark(
                new int[] {0, 0, 0, 0, 1},
                progressTracker,
                recordingProgressListener,
                new int[] {0, 1, 1, 0, 1},
                new String[] {"0-0", "1-0", "1-1", "0-1", "1-2"},
                2);
        assertEquals(Collections.singletonList(2), recordingProgressListener.notifications);

        recordingProgressListener.reset();
        testOnEpochWatermark(
                new int[] {0, 0, 0, 0, 1},
                progressTracker,
                recordingProgressListener,
                new int[] {0, 0, 1, 1, 1},
                new String[] {"0-0", "0-1", "1-0", "1-1", "1-2"},
                3);
        assertEquals(Collections.singletonList(3), recordingProgressListener.notifications);
    }

    private void testOnEpochWatermark(
            int[] expectedNumNotifications,
            OperatorEpochWatermarkTracker tracker,
            RecordingProgressListener recordingProgressListener,
            int[] inputIndices,
            String[] senders,
            int incrementedEpochWatermark)
            throws IOException {
        for (int i = 0; i < expectedNumNotifications.length; ++i) {
            tracker.onEpochWatermark(inputIndices[i], senders[i], incrementedEpochWatermark);
            assertEquals(
                    expectedNumNotifications[i], recordingProgressListener.notifications.size());
        }
    }

    private static class RecordingProgressListener
            implements OperatorEpochWatermarkTrackerListener {

        final List<Integer> notifications = new ArrayList<>();

        @Override
        public void onEpochWatermarkIncrement(int epochWatermark) {
            notifications.add(epochWatermark);
        }

        public void reset() {
            notifications.clear();
        }
    }
}
