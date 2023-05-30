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

package org.apache.flink.ml.common.ps.training;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;

import org.apache.commons.collections.IteratorUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/** The ML session for machine learning algorithms that adopts mini-batch training. */
public class MiniBatchMLSession<DT> extends MLSessionImpl<DT> {

    /** The placeholder for indices to pull for each iteration. */
    public long[] pullIndices;
    /** The placeholder for the pulled values for each iteration. */
    public double[] pulledValues;
    /** The placeholder for indices to push for each iteration. */
    public long[] pushIndices;
    /** The placeholder for values to push for each iteration. */
    public double[] pushValues;

    /** The batch of training data for computing gradients. */
    public List<DT> batchData;

    private ListState<DT> batchDataState;
    /** Global batch size. */
    private final int globalBatchSize;
    /** The local batch size. */
    private int localBatchSize;
    /** Type information of the input data. */
    private final TypeInformation<DT> typeInformation;

    public MiniBatchMLSession(int globalBatchSize, TypeInformation<DT> typeInformation) {
        this.globalBatchSize = globalBatchSize;
        this.typeInformation = typeInformation;
    }

    @Override
    public void setWorldInfo(int workerId, int numWorkers) {
        super.setWorldInfo(workerId, numWorkers);
        this.localBatchSize = globalBatchSize / numWorkers;
        if (globalBatchSize % numWorkers > workerId) {
            localBatchSize++;
        }
        this.batchData = new ArrayList<>(localBatchSize);
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        batchDataState =
                context.getOperatorStateStore()
                        .getListState(new ListStateDescriptor<>("batchDataState", typeInformation));

        Iterator<DT> batchDataIterator = batchDataState.get().iterator();
        if (batchDataIterator.hasNext()) {
            batchData = IteratorUtils.toList(batchDataIterator);
        }
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        batchDataState.clear();
        if (batchData.size() > 0) {
            batchDataState.addAll(batchData);
        }
    }

    /** Reads in next batch of training data. */
    public void readInNextBatchData() throws IOException {
        batchData.clear();
        int i = 0;
        while (i < localBatchSize && inputData.hasNext()) {
            batchData.add(inputData.next());
            i++;
        }
        if (!inputData.hasNext()) {
            inputData.reset();
        }
    }
}
