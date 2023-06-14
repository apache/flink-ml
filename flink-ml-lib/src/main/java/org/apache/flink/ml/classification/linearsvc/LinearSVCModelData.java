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

package org.apache.flink.ml.classification.linearsvc;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorSerializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Model data of {@link LinearSVCModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class LinearSVCModelData {

    public DenseIntDoubleVector coefficient;

    public LinearSVCModelData(DenseIntDoubleVector coefficient) {
        this.coefficient = coefficient;
    }

    public LinearSVCModelData() {}

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<LinearSVCModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(
                        x ->
                                new LinearSVCModelData(
                                        (DenseIntDoubleVector)
                                                (((Vector<Integer, Double, int[], double[]>)
                                                                x.getField(0))
                                                        .toDense())));
    }

    /** Data encoder for {@link LinearSVCModel}. */
    public static class ModelDataEncoder implements Encoder<LinearSVCModelData> {
        private final DenseIntDoubleVectorSerializer serializer =
                new DenseIntDoubleVectorSerializer();

        @Override
        public void encode(LinearSVCModelData modelData, OutputStream outputStream)
                throws IOException {
            serializer.serialize(
                    modelData.coefficient, new DataOutputViewStreamWrapper(outputStream));
        }
    }

    /** Data decoder for {@link LinearSVCModel}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<LinearSVCModelData> {

        @Override
        public Reader<LinearSVCModelData> createReader(
                Configuration configuration, FSDataInputStream inputStream) {
            return new Reader<LinearSVCModelData>() {
                private final DenseIntDoubleVectorSerializer serializer =
                        new DenseIntDoubleVectorSerializer();

                @Override
                public LinearSVCModelData read() throws IOException {
                    try {
                        DenseIntDoubleVector coefficient =
                                serializer.deserialize(new DataInputViewStreamWrapper(inputStream));
                        return new LinearSVCModelData(coefficient);
                    } catch (EOFException e) {
                        return null;
                    }
                }

                @Override
                public void close() throws IOException {
                    inputStream.close();
                }
            };
        }

        @Override
        public TypeInformation<LinearSVCModelData> getProducedType() {
            return TypeInformation.of(LinearSVCModelData.class);
        }
    }
}
