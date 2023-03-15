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

import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

/** Model data of {@link LogisticRegressionModelServable}. */
public class LogisticRegressionModelData {

    public DenseVector coefficient;

    public long modelVersion;

    public LogisticRegressionModelData() {}

    public LogisticRegressionModelData(DenseVector coefficient) {
        this.coefficient = coefficient;
    }

    public LogisticRegressionModelData(DenseVector coefficient, long modelVersion) {
        this(coefficient);
        this.modelVersion = modelVersion;
    }

    /**
     * Serializes the model data into byte array which can be saved to external storage and then be
     * used to update the servable by `TransformerServable::setModelData` method.
     *
     * @return The serialized model data in byte array.
     */
    public byte[] serialize() throws IOException {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

        DataOutputViewStreamWrapper outputViewStreamWrapper =
                new DataOutputViewStreamWrapper(outputStream);

        DenseVectorSerializer serializer = new DenseVectorSerializer();
        serializer.serialize(coefficient, outputViewStreamWrapper);
        outputViewStreamWrapper.writeLong(modelVersion);

        return outputStream.toByteArray();
    }
}
