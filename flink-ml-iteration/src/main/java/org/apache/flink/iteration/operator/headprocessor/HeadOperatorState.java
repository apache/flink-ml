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

package org.apache.flink.iteration.operator.headprocessor;

import java.util.Collections;
import java.util.Map;

/** The state entry for the head operator. */
public class HeadOperatorState {

    public static final HeadOperatorState FINISHED_STATE =
            new HeadOperatorState(Collections.emptyMap(), 0, 0);

    private Map<Integer, Long> numFeedbackRecordsEachRound;

    private int latestRoundAligned;

    private int latestRoundGloballyAligned;

    public HeadOperatorState(
            Map<Integer, Long> numFeedbackRecordsEachRound,
            int latestRoundAligned,
            int latestRoundGloballyAligned) {
        this.numFeedbackRecordsEachRound = numFeedbackRecordsEachRound;
        this.latestRoundAligned = latestRoundAligned;
        this.latestRoundGloballyAligned = latestRoundGloballyAligned;
    }

    public Map<Integer, Long> getNumFeedbackRecordsEachRound() {
        return numFeedbackRecordsEachRound;
    }

    public int getLatestRoundAligned() {
        return latestRoundAligned;
    }

    public int getLatestRoundGloballyAligned() {
        return latestRoundGloballyAligned;
    }
}
