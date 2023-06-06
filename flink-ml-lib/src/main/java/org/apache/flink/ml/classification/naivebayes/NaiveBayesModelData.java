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
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.common.typeutils.base.MapSerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorSerializer;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
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
public class NaiveBayesModelData {

    private static final Map<String, TypeInformation<?>> fields;

    static {
        fields = new HashMap<>();
        fields.put(
                "theta",
                Types.OBJECT_ARRAY(Types.OBJECT_ARRAY(Types.MAP(Types.DOUBLE, Types.DOUBLE))));
        fields.put("piArray", DenseIntDoubleVectorTypeInfo.INSTANCE);
        fields.put("labels", DenseIntDoubleVectorTypeInfo.INSTANCE);
    }

    public static final TypeInformation<NaiveBayesModelData> TYPE_INFO =
            Types.POJO(NaiveBayesModelData.class, fields);

    /**
     * Log of class conditional probabilities, whose dimension is C (number of classes) by D (number
     * of features).
     */
    public Map<Double, Double>[][] theta;

    /** Log of class priors, whose dimension is C (number of classes). */
    public DenseIntDoubleVector piArray;

    /** Value of labels. */
    public DenseIntDoubleVector labels;

    public NaiveBayesModelData(
            Map<Double, Double>[][] theta,
            DenseIntDoubleVector piArray,
            DenseIntDoubleVector labels) {
        this.theta = theta;
        this.piArray = piArray;
        this.labels = labels;
    }

    public NaiveBayesModelData() {}

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
                                                ((IntDoubleVector) row.getField(1)).toDense(),
                                                ((IntDoubleVector) row.getField(2)).toDense()),
                        TYPE_INFO);
    }

    /** Data encoder for the {@link NaiveBayesModelData}. */
    public static class ModelDataEncoder implements Encoder<NaiveBayesModelData> {
        private final DenseIntDoubleVectorSerializer serializer =
                new DenseIntDoubleVectorSerializer();

        @Override
        public void encode(NaiveBayesModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputViewStreamWrapper outputViewStreamWrapper =
                    new DataOutputViewStreamWrapper(outputStream);

            MapSerializer<Double, Double> mapSerializer =
                    new MapSerializer<>(DoubleSerializer.INSTANCE, DoubleSerializer.INSTANCE);

            serializer.serialize(modelData.labels, outputViewStreamWrapper);

            serializer.serialize(modelData.piArray, outputViewStreamWrapper);

            outputViewStreamWrapper.writeInt(modelData.theta.length);
            outputViewStreamWrapper.writeInt(modelData.theta[0].length);
            for (Map<Double, Double>[] maps : modelData.theta) {
                for (Map<Double, Double> map : maps) {
                    mapSerializer.serialize(map, outputViewStreamWrapper);
                }
            }
        }
    }

    /** Data decoder for the {@link NaiveBayesModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<NaiveBayesModelData> {
        @Override
        public Reader<NaiveBayesModelData> createReader(
                Configuration config, FSDataInputStream inputStream) {
            return new Reader<NaiveBayesModelData>() {
                private final DenseIntDoubleVectorSerializer serializer =
                        new DenseIntDoubleVectorSerializer();

                @Override
                public NaiveBayesModelData read() throws IOException {
                    try {
                        DataInputViewStreamWrapper inputViewStreamWrapper =
                                new DataInputViewStreamWrapper(inputStream);
                        MapSerializer<Double, Double> mapSerializer =
                                new MapSerializer<>(
                                        DoubleSerializer.INSTANCE, DoubleSerializer.INSTANCE);

                        DenseIntDoubleVector labels =
                                serializer.deserialize(inputViewStreamWrapper);

                        DenseIntDoubleVector piArray =
                                serializer.deserialize(inputViewStreamWrapper);

                        int featureSize = inputViewStreamWrapper.readInt();
                        int numLabels = inputViewStreamWrapper.readInt();
                        Map<Double, Double>[][] theta = new HashMap[numLabels][featureSize];
                        for (int i = 0; i < featureSize; i++) {
                            for (int j = 0; j < numLabels; j++) {
                                theta[i][j] = mapSerializer.deserialize(inputViewStreamWrapper);
                            }
                        }
                        return new NaiveBayesModelData(theta, piArray, labels);
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
            return TYPE_INFO;
        }
    }
}
