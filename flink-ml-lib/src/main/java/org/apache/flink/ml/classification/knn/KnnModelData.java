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

package org.apache.flink.ml.classification.knn;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.api.ModelInfo;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseMatrixSerializer;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

/**
 * Model data of {@link KnnModel}.
 *
 * <p>This class also provides methods to convert model data from Table to a data stream, and
 * classes to save/load model data.
 */
public class KnnModelData {
    public DenseMatrix packedFeatures;
    public DenseVector featureNormSquares;
    public DenseVector labels;

    public KnnModelData() {}

    public KnnModelData(List<KnnModelElement> modelRecords) {
        int featureDim = modelRecords.get(0).packedFeatures.numRows();
        int totalCols = 0;
        for (KnnModelElement record : modelRecords) {
            totalCols += record.packedFeatures.numCols();
        }
        packedFeatures = new DenseMatrix(featureDim, totalCols);
        featureNormSquares = new DenseVector(totalCols);
        labels = new DenseVector(totalCols);
        int offset = 0;
        for (KnnModelElement record : modelRecords) {
            System.arraycopy(
                    record.packedFeatures.values,
                    0,
                    packedFeatures.values,
                    offset * featureDim,
                    featureDim * record.packedFeatures.numCols());
            System.arraycopy(
                    record.featureNormSquares.values,
                    0,
                    featureNormSquares.values,
                    offset,
                    record.featureNormSquares.size());
            System.arraycopy(record.labels.values, 0, labels.values, offset, record.labels.size());
            offset += record.featureNormSquares.size();
        }
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelDataTable The table model data.
     * @return The data stream model data.
     */
    public static DataStream<KnnModelElement> getModelDataStream(Table modelDataTable) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        return tEnv.toDataStream(modelDataTable).map(x -> (KnnModelElement) x.getField(0));
    }

    /** Encoder for {@link KnnModelData}. */
    public static class ModelDataEncoder implements Encoder<KnnModelElement> {
        @Override
        public void encode(KnnModelElement modelRecord, OutputStream outputStream)
                throws IOException {
            DataOutputView dataOutputView = new DataOutputViewStreamWrapper(outputStream);
            DenseMatrixSerializer.INSTANCE.serialize(modelRecord.packedFeatures, dataOutputView);
            DenseVectorSerializer.INSTANCE.serialize(
                    modelRecord.featureNormSquares, dataOutputView);
            DenseVectorSerializer.INSTANCE.serialize(modelRecord.labels, dataOutputView);
            dataOutputView.writeLong(modelRecord.versionId);
            dataOutputView.writeBoolean(modelRecord.isLastRecord);
        }
    }

    /** Decoder for {@link KnnModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<KnnModelElement> {
        @Override
        public Reader<KnnModelElement> createReader(Configuration config, FSDataInputStream stream) {
            return new Reader<KnnModelElement>() {

                private final DataInputView source = new DataInputViewStreamWrapper(stream);

                @Override
                public KnnModelElement read() throws IOException {
                    try {
                        DenseMatrix matrix = DenseMatrixSerializer.INSTANCE.deserialize(source);
                        DenseVector normSquares =
                                DenseVectorSerializer.INSTANCE.deserialize(source);
                        DenseVector labels = DenseVectorSerializer.INSTANCE.deserialize(source);
                        return new KnnModelElement(
                                matrix,
                                normSquares,
                                labels,
                                source.readLong(),
                                source.readBoolean());
                    } catch (EOFException e) {
                        return null;
                    }
                }

                @Override
                public void close() throws IOException {
                    stream.close();
                }
            };
        }

        @Override
        public TypeInformation<KnnModelElement> getProducedType() {
            return TypeInformation.of(KnnModelElement.class);
        }
    }

    public static class KnnModelElement implements ModelInfo {
        public DenseMatrix packedFeatures;
        public DenseVector featureNormSquares;
        public DenseVector labels;
        public long versionId;
        public boolean isLastRecord;

        public KnnModelElement(
                DenseMatrix packedFeatures,
                DenseVector featureNormSquares,
                DenseVector labels,
                long versionId,
                boolean isLastRecord) {
            this.packedFeatures = packedFeatures;
            this.featureNormSquares = featureNormSquares;
            this.labels = labels;
            this.versionId = versionId;
            this.isLastRecord = isLastRecord;
        }

        @Override
        public long getVersionId() {
            return versionId;
        }

        @Override
        public boolean getIsLastRecord() {
            return isLastRecord;
        }
    }
}
