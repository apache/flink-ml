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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** Model data of {@link LogisticRegressionModelServable}. */
public class LogisticRegressionModelData {

    public DenseIntDoubleVector coefficient;

    public long modelVersion;

    public LogisticRegressionModelData() {}

    public LogisticRegressionModelData(DenseIntDoubleVector coefficient, long modelVersion) {
        this.coefficient = coefficient;
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
        long modelVersion = dataInputViewStreamWrapper.readLong();

        return new LogisticRegressionModelData(coefficient, modelVersion);
    }
}
