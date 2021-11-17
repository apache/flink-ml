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
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
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
 * The model data of {@link NaiveBayesModel}.
 *
 * <p>This class also provides methods to convert model data between Table and Datastream, and
 * classes to save/load model data.
 */
public class NaiveBayesModelData {
    /**
     * Log of class conditional probabilities, whose dimension is C (number of classes) by D (number
     * of features).
     */
    public final Map<Double, Double>[][] theta;

    /** Log of class priors, whose dimension is C (number of classes). */
    public final DenseVector piArray;

    /** Value of labels. */
    public final DenseVector labels;

    public NaiveBayesModelData(Map<Double, Double>[][] theta, double[] piArray, double[] labels) {
        this(theta, Vectors.dense(piArray), Vectors.dense(labels));
    }

    public NaiveBayesModelData(
            Map<Double, Double>[][] theta, DenseVector piArray, DenseVector labels) {
        this.theta = theta;
        this.piArray = piArray;
        this.labels = labels;
    }

    /** Converts the provided modelData Datastream into corresponding Table. */
    public static Table getModelDataTable(DataStream<NaiveBayesModelData> stream) {
        StreamTableEnvironment tEnv =
                StreamTableEnvironment.create(stream.getExecutionEnvironment());
        return tEnv.fromDataStream(stream);
    }

    /** Converts the provided modelData Table into corresponding DataStream. */
    public static DataStream<NaiveBayesModelData> getModelDataStream(Table table) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) table).getTableEnvironment();
        return tEnv.toDataStream(table)
                .map(
                        (MapFunction<Row, NaiveBayesModelData>)
                                row -> (NaiveBayesModelData) row.getField("f0"));
    }

    /** Encoder for the {@link NaiveBayesModelData}. */
    public static class ModelDataEncoder implements Encoder<NaiveBayesModelData> {
        @Override
        public void encode(NaiveBayesModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputViewStreamWrapper output = new DataOutputViewStreamWrapper(outputStream);

            DenseVectorSerializer denseVectorSerializer = new DenseVectorSerializer();
            MapSerializer<Double, Double> mapSerializer =
                    new MapSerializer<>(new DoubleSerializer(), new DoubleSerializer());

            denseVectorSerializer.serialize(modelData.labels, output);

            denseVectorSerializer.serialize(modelData.piArray, output);

            output.writeInt(modelData.theta.length);
            output.writeInt(modelData.theta[0].length);
            for (Map<Double, Double>[] maps : modelData.theta) {
                for (Map<Double, Double> map : maps) {
                    mapSerializer.serialize(map, output);
                }
            }
        }
    }

    /** Decoder for the {@link NaiveBayesModelData}. */
    public static class ModelDataStreamFormat extends SimpleStreamFormat<NaiveBayesModelData> {
        @Override
        public Reader<NaiveBayesModelData> createReader(
                Configuration config, FSDataInputStream inputStream) {
            return new Reader<NaiveBayesModelData>() {
                private final DataInputViewStreamWrapper input =
                        new DataInputViewStreamWrapper(inputStream);

                @Override
                public NaiveBayesModelData read() throws IOException {
                    try {
                        DenseVectorSerializer denseVectorSerializer = new DenseVectorSerializer();
                        MapSerializer<Double, Double> mapSerializer =
                                new MapSerializer<>(new DoubleSerializer(), new DoubleSerializer());

                        DenseVector labels = denseVectorSerializer.deserialize(input);

                        DenseVector piArray = denseVectorSerializer.deserialize(input);

                        int featureSize = input.readInt();
                        int numLabels = input.readInt();
                        Map<Double, Double>[][] theta = new HashMap[numLabels][featureSize];
                        for (int i = 0; i < featureSize; i++) {
                            for (int j = 0; j < numLabels; j++) {
                                theta[i][j] = mapSerializer.deserialize(input);
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
            return TypeInformation.of(NaiveBayesModelData.class);
        }
    }
}
