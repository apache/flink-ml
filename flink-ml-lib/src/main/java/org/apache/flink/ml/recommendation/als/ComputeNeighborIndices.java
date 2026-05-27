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
import org.apache.flink.ml.recommendation.als.Als.Ratings;
import org.apache.flink.ml.recommendation.als.AlsMLSession.BlockData;

/** An iteration stage that computes the indices needed to update factors. */
public class ComputeNeighborIndices extends ProcessStage<AlsMLSession> {

    private final int rank;

    public ComputeNeighborIndices(int rank) {
        this.rank = rank;
    }

    @Override
    public void process(AlsMLSession session) throws Exception {
        session.log(this.getClass().getSimpleName(), true);

        session.prepareNextRatingsBatchData();

        BlockData blockData = session.batchData;

        session.reusedNeighborsSet.clear();
        session.reusedNeighborIndexMapping.clear();
        session.pullIndices.clear();

        for (Ratings ratings : blockData.ratingsList) {
            for (long neighbor : ratings.neighbors) {
                session.reusedNeighborsSet.add(neighbor);
            }
        }

        if (session.reusedNeighborsSet.size() == 0) {
            session.pullIndices.add(Long.MIN_VALUE);
        } else {
            int it = 0;
            for (Long aLong : session.reusedNeighborsSet) {
                session.pullIndices.add(aLong);
                session.reusedNeighborIndexMapping.put(aLong.longValue(), it++);
            }
        }
        session.pullValues.size(session.pullIndices.size() * rank);
        session.log(this.getClass().getSimpleName(), false);
    }
}
