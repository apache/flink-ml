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
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.NormalEquationSolver;

import java.io.IOException;
import java.util.Arrays;

/**
 * An iteration stage that uses the pulled least square matrices and vector data to compute the
 * factors.
 */
public class UpdateHotPointFactors extends ProcessStage<AlsMLSession> {
    private final boolean nonNegative;
    private final int rank;

    public UpdateHotPointFactors(int rank, boolean nonNegative) {
        this.rank = rank;
        this.nonNegative = nonNegative;
    }

    @Override
    public void process(AlsMLSession session) throws IOException {
        session.log(this.getClass().getSimpleName(), true);
        SharedLongArray indices = session.pullIndices;
        SharedDoubleArray modelValues = session.pullValues;
        session.pushIndices.clear();
        session.pushValues.clear();
        if (session.batchData.numSplitNodeIds != 0) {
            session.pushIndices.size(session.batchData.numSplitNodeIds);
            session.pushValues.size(session.batchData.numSplitNodeIds * rank);
        } else {
            session.pushValues.addAll(new double[rank]);
            session.pushIndices.add(Long.MIN_VALUE);
            return;
        }
        int offset = rank * (rank + 1);
        for (int i = 0; i < indices.size(); ++i) {
            if (indices.get(i) == Long.MIN_VALUE) {
                session.pushIndices.add(Long.MIN_VALUE);
                continue;
            }
            DenseVector x = new DenseVector(rank);

            DenseMatrix ata =
                    new DenseMatrix(
                            rank,
                            rank,
                            Arrays.copyOfRange(
                                    modelValues.elements(), i * offset, i * offset + rank * rank));
            DenseVector atb =
                    new DenseVector(
                            Arrays.copyOfRange(
                                    modelValues.elements(),
                                    i * offset + rank * rank,
                                    (i + 1) * offset));
            NormalEquationSolver ls = new NormalEquationSolver(rank, ata, atb);
            ls.solve(x, nonNegative);
            System.arraycopy(x.values, 0, session.pushValues.elements(), i * rank, rank);

            if (session.pullIndices.get(i) != Long.MIN_VALUE) {
                session.pushIndices.elements()[i] = -session.pullIndices.elements()[i] - 1;
            } else {
                session.pushIndices.elements()[i] = session.pullIndices.elements()[i];
            }
        }
        AlsMLSession.LOG.info(
                String.format(
                        "Worker-%d hot point push size %d",
                        session.workerId, session.pushIndices.size()));
        session.log(this.getClass().getSimpleName(), false);
    }
}
