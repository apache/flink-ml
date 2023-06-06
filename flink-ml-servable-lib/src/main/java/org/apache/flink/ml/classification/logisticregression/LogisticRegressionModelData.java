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

package org.apache.flink.ml.classification.logisticregression;

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorSerializer;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

/** Model data of {@link LogisticRegressionModelServable}. */
public class LogisticRegressionModelData {

    public DenseIntDoubleVector coefficient;

    public long startIndex;

    public long endIndex;

    public long modelVersion;

    public LogisticRegressionModelData() {}

    public LogisticRegressionModelData(DenseIntDoubleVector coefficient, long modelVersion) {
        this(coefficient, 0L, coefficient.size(), modelVersion);
    }

    public LogisticRegressionModelData(
            DenseIntDoubleVector coefficient, long startIndex, long endIndex, long modelVersion) {
        this.coefficient = coefficient;
        this.startIndex = startIndex;
        this.endIndex = endIndex;
        this.modelVersion = modelVersion;
    }

    /**
     * Serializes the instance and writes to the output stream.
     *
     * @param outputStream The stream to write to.
     */
    @VisibleForTesting
    public void encode(OutputStream outputStream) throws IOException {
        DataOutputViewStreamWrapper dataOutputViewStreamWrapper =
                new DataOutputViewStreamWrapper(outputStream);

        DenseIntDoubleVectorSerializer serializer = new DenseIntDoubleVectorSerializer();
        serializer.serialize(coefficient, dataOutputViewStreamWrapper);
        dataOutputViewStreamWrapper.writeLong(startIndex);
        dataOutputViewStreamWrapper.writeLong(endIndex);
        dataOutputViewStreamWrapper.writeLong(modelVersion);
    }

    /**
     * Reads and deserializes the model data from the input stream.
     *
     * @param inputStream The stream to read from.
     * @return The model data instance.
     */
    static LogisticRegressionModelData decode(InputStream inputStream) throws IOException {
        DataInputViewStreamWrapper dataInputViewStreamWrapper =
                new DataInputViewStreamWrapper(inputStream);

        DenseIntDoubleVectorSerializer serializer = new DenseIntDoubleVectorSerializer();
        DenseIntDoubleVector coefficient = serializer.deserialize(dataInputViewStreamWrapper);
        long startIndex = dataInputViewStreamWrapper.readLong();
        long endIndex = dataInputViewStreamWrapper.readLong();
        long modelVersion = dataInputViewStreamWrapper.readLong();

        return new LogisticRegressionModelData(coefficient, startIndex, endIndex, modelVersion);
    }

    @VisibleForTesting
    public static LogisticRegressionModelData mergeSegments(
            List<LogisticRegressionModelData> segments) {
        long dim = 0;
        for (LogisticRegressionModelData segment : segments) {
            dim = Math.max(dim, segment.endIndex);
        }
        // TODO: Add distributed inference for very large models.
        Preconditions.checkState(
                dim < Integer.MAX_VALUE,
                "The dimension of logistic regression model is larger than INT.MAX. Please consider using distributed inference.");
        int intDim = (int) dim;
        DenseIntDoubleVector mergedCoefficient = new DenseIntDoubleVector(intDim);
        for (LogisticRegressionModelData segment : segments) {
            int startIndex = (int) segment.startIndex;
            int endIndex = (int) segment.endIndex;
            System.arraycopy(
                    segment.coefficient.values,
                    0,
                    mergedCoefficient.values,
                    startIndex,
                    endIndex - startIndex);
        }
        return new LogisticRegressionModelData(
                mergedCoefficient, 0, mergedCoefficient.size(), segments.get(0).modelVersion);
    }
}
