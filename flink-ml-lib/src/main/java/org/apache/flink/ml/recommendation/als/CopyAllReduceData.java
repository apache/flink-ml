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

/** An iteration stage that copy the aggregating data to the all reduce data. */
public class CopyAllReduceData extends ProcessStage<AlsMLSession> {

    private final int rank;

    public CopyAllReduceData(int rank) {
        this.rank = rank;
    }

    @Override
    public void process(AlsMLSession session) throws Exception {
        System.arraycopy(
                session.aggregatorSDAArray.elements(),
                0,
                session.allReduceBuffer[0],
                0,
                rank * rank);
    }
}
