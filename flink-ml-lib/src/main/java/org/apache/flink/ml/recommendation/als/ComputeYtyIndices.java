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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.ml.common.ps.iterations.ProcessStage;

/** An iteration stage that calculates the indices for yty matrix computing. */
public class ComputeYtyIndices extends ProcessStage<AlsMLSession> {

    @Override
    public void process(AlsMLSession session) throws Exception {
        session.log(this.getClass().getSimpleName(), true);
        if (!session.isRatingsInitialized) {
            session.initializeRatingsBatchData();
            session.isRatingsInitialized = true;
        }
        session.pullIndices.clear();
        session.pullValues.clear();

        if (session.updateUserFactors) {
            if (session.itemIds.length == 0 || session.currentItemIndex != 0) {
                session.pullIndices.addAll(new long[] {Long.MIN_VALUE});
            } else {
                session.pullIndices.addAll(session.itemIds);
            }
        } else {
            if (session.userIds.length == 0 || session.currentUserIndex != 0) {
                session.pullIndices.addAll(new long[] {Long.MIN_VALUE});
            } else {
                session.pullIndices.addAll(session.userIds);
            }
        }
        session.log(this.getClass().getSimpleName(), false);
    }
}
