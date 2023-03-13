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

package org.apache.flink.ml.feature.idf;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * An Estimator that computes the inverse document frequency (IDF) for the input documents. IDF is
 * computed following `idf = log((m + 1) / (d(t) + 1))`, where `m` is the total number of documents
 * and `d(t)` is the number of documents that contains `t`.
 *
 * <p>Users could filter out terms that appeared in little documents by setting {@link
 * IDFParams#getMinDocFreq()}.
 *
 * <p>See https://en.wikipedia.org/wiki/Tf%E2%80%93idf.
 */
public class IDF implements Estimator<IDF, IDFModel>, IDFParams<IDF> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public IDF() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public IDFModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        final String inputCol = getInputCol();
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<Vector> inputData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, Vector>)
                                        value -> ((Vector) value.getField(inputCol)),
                                VectorTypeInfo.INSTANCE);

        DataStream<IDFModelData> modelData =
                DataStreamUtils.aggregate(inputData, new IDFAggregator(getMinDocFreq()));

        IDFModel model = new IDFModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static IDF load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    /** The main logic to compute the model data of IDF. */
    private static class IDFAggregator
            implements AggregateFunction<Vector, Tuple2<Long, DenseVector>, IDFModelData> {
        private final int minDocFreq;

        public IDFAggregator(int minDocFreq) {
            this.minDocFreq = minDocFreq;
        }

        @Override
        public Tuple2<Long, DenseVector> createAccumulator() {
            return Tuple2.of(0L, new DenseVector(new double[0]));
        }

        @Override
        public Tuple2<Long, DenseVector> add(
                Vector vector, Tuple2<Long, DenseVector> numDocsAndDocFreq) {
            if (numDocsAndDocFreq.f0 == 0) {
                numDocsAndDocFreq.f1 = new DenseVector(vector.size());
            }
            numDocsAndDocFreq.f0 += 1L;

            double[] values;
            if (vector instanceof SparseVector) {
                values = ((SparseVector) vector).values;
            } else {
                values = ((DenseVector) vector).values;
            }
            for (int i = 0; i < values.length; i++) {
                values[i] = values[i] > 0 ? 1 : 0;
            }

            BLAS.axpy(1, vector, numDocsAndDocFreq.f1);

            return numDocsAndDocFreq;
        }

        @Override
        public IDFModelData getResult(Tuple2<Long, DenseVector> numDocsAndDocFreq) {
            long numDocs = numDocsAndDocFreq.f0;
            DenseVector docFreq = numDocsAndDocFreq.f1;
            Preconditions.checkState(numDocs > 0, "The training set is empty.");

            long[] filteredDocFreq = new long[docFreq.size()];
            double[] df = docFreq.values;
            double[] idf = new double[df.length];
            for (int i = 0; i < idf.length; i++) {
                if (df[i] >= minDocFreq) {
                    idf[i] = Math.log((numDocs + 1) / (df[i] + 1));
                    filteredDocFreq[i] = (long) df[i];
                }
            }
            return new IDFModelData(Vectors.dense(idf), filteredDocFreq, numDocs);
        }

        @Override
        public Tuple2<Long, DenseVector> merge(
                Tuple2<Long, DenseVector> numDocsAndDocFreq1,
                Tuple2<Long, DenseVector> numDocsAndDocFreq2) {
            if (numDocsAndDocFreq1.f0 == 0) {
                return numDocsAndDocFreq2;
            }

            if (numDocsAndDocFreq2.f0 == 0) {
                return numDocsAndDocFreq1;
            }

            numDocsAndDocFreq2.f0 += numDocsAndDocFreq1.f0;
            BLAS.axpy(1, numDocsAndDocFreq1.f1, numDocsAndDocFreq2.f1);
            return numDocsAndDocFreq2;
        }
    }
}
