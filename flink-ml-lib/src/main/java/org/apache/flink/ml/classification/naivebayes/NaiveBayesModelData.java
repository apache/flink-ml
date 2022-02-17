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

package org.apache.flink.ml.classification.naivebayes;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.common.typeutils.base.MapSerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.api.ModelInfo;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Model data of {@link NaiveBayesModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class NaiveBayesModelData implements ModelInfo {
    /**
     * Log of class conditional probabilities, whose dimension is C (number of classes) by D (number
     * of features).
     */
    public Map<Double, Double>[][] theta;

    /** Log of class priors, whose dimension is C (number of classes). */
    public DenseVector piArray;

    /** Value of labels. */
    public DenseVector labels;

    public long versionId;
    public boolean isLastRecord;

    public NaiveBayesModelData(
            Map<Double, Double>[][] theta, DenseVector piArray, DenseVector labels) {
        this(theta, piArray, labels, System.nanoTime(), true);
    }

    public NaiveBayesModelData(
            Map<Double, Double>[][] theta,
            DenseVector piArray,
            DenseVector labels,
            long versionId,
            boolean isLastRecord) {
        this.theta = theta;
        this.piArray = piArray;
        this.labels = labels;
        this.versionId = versionId;
        this.isLastRecord = isLastRecord;
    }

    public NaiveBayesModelData() {}

    public NaiveBayesModelData setVersionId(long versionId) {
        this.versionId = versionId;
        return this;
    }

    @Override
    public long getVersionId() {
        return this.versionId;
    }

    @Override
    public boolean getIsLastRecord() {
        return isLastRecord;
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<NaiveBayesModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(
                        (MapFunction<Row, NaiveBayesModelData>)
                                row ->
                                        new NaiveBayesModelData(
                                                (Map<Double, Double>[][]) row.getField(0),
                                                (DenseVector) row.getField(1),
                                                (DenseVector) row.getField(2),
                                                (long) row.getField(3),
                                                (boolean) row.getField(4)));
    }

    /** Data encoder for the {@link NaiveBayesModelData}. */
    public static class ModelDataEncoder implements Encoder<NaiveBayesModelData> {
        @Override
        public void encode(NaiveBayesModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputViewStreamWrapper outputViewStreamWrapper =
                    new DataOutputViewStreamWrapper(outputStream);

            MapSerializer<Double, Double> mapSerializer =
                    new MapSerializer<>(DoubleSerializer.INSTANCE, DoubleSerializer.INSTANCE);

            DenseVectorSerializer.INSTANCE.serialize(modelData.labels, outputViewStreamWrapper);

            DenseVectorSerializer.INSTANCE.serialize(modelData.piArray, outputViewStreamWrapper);

            outputViewStreamWrapper.writeInt(modelData.theta.length);
            outputViewStreamWrapper.writeInt(modelData.theta[0].length);
            for (Map<Double, Double>[] maps : modelData.theta) {
                for (Map<Double, Double> map : maps) {
                    mapSerializer.serialize(map, outputViewStreamWrapper);
                }
            }
            outputViewStreamWrapper.writeLong(modelData.versionId);
            outputViewStreamWrapper.writeBoolean(modelData.isLastRecord);
        }
    }

    /** Data decoder for the {@link NaiveBayesModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<NaiveBayesModelData> {
        @Override
        public Reader<NaiveBayesModelData> createReader(
                Configuration config, FSDataInputStream inputStream) {
            return new Reader<NaiveBayesModelData>() {

                @Override
                public NaiveBayesModelData read() throws IOException {
                    try {
                        DataInputViewStreamWrapper inputViewStreamWrapper =
                                new DataInputViewStreamWrapper(inputStream);
                        MapSerializer<Double, Double> mapSerializer =
                                new MapSerializer<>(
                                        DoubleSerializer.INSTANCE, DoubleSerializer.INSTANCE);

                        DenseVector labels =
                                DenseVectorSerializer.INSTANCE.deserialize(inputViewStreamWrapper);

                        DenseVector piArray =
                                DenseVectorSerializer.INSTANCE.deserialize(inputViewStreamWrapper);

                        int featureSize = inputViewStreamWrapper.readInt();
                        int numLabels = inputViewStreamWrapper.readInt();
                        Map<Double, Double>[][] theta = new HashMap[numLabels][featureSize];
                        for (int i = 0; i < featureSize; i++) {
                            for (int j = 0; j < numLabels; j++) {
                                theta[i][j] = mapSerializer.deserialize(inputViewStreamWrapper);
                            }
                        }
                        return new NaiveBayesModelData(
                                theta,
                                piArray,
                                labels,
                                inputViewStreamWrapper.readLong(),
                                inputViewStreamWrapper.readBoolean());
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
        public TypeInformation<NaiveBayesModelData> getProducedType() {
            return TypeInformation.of(NaiveBayesModelData.class);
        }
    }
}
