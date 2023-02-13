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

package org.apache.flink.ml.common.gbt;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.classification.gbtclassifier.GBTClassifierModel;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.LocalState;
import org.apache.flink.ml.common.gbt.defs.Node;
import org.apache.flink.ml.common.gbt.typeinfo.GBTModelDataSerializer;
import org.apache.flink.ml.common.gbt.typeinfo.GBTModelDataTypeInfoFactory;
import org.apache.flink.ml.feature.stringindexer.StringIndexerModel;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.regression.gbtregressor.GBTRegressorModel;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;

import org.eclipse.collections.impl.map.mutable.primitive.IntDoubleHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectIntHashMap;

import java.io.IOException;
import java.io.OutputStream;
import java.util.BitSet;
import java.util.List;

/**
 * Model data of {@link GBTClassifierModel} and {@link GBTRegressorModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
@TypeInfo(GBTModelDataTypeInfoFactory.class)
public class GBTModelData {

    public String type;
    public boolean isInputVector;

    public double prior;
    public double stepSize;

    public List<Node> roots;
    public IntObjectHashMap<ObjectIntHashMap<String>> categoryToIdMaps;
    public IntObjectHashMap<double[]> featureIdToBinEdges;
    public BitSet isCategorical;

    public GBTModelData() {}

    public GBTModelData(
            String type,
            boolean isInputVector,
            double prior,
            double stepSize,
            List<Node> roots,
            IntObjectHashMap<ObjectIntHashMap<String>> categoryToIdMaps,
            IntObjectHashMap<double[]> featureIdToBinEdges,
            BitSet isCategorical) {
        this.type = type;
        this.isInputVector = isInputVector;
        this.prior = prior;
        this.stepSize = stepSize;
        this.roots = roots;
        this.categoryToIdMaps = categoryToIdMaps;
        this.featureIdToBinEdges = featureIdToBinEdges;
        this.isCategorical = isCategorical;
    }

    public static GBTModelData fromLocalState(LocalState state) {
        IntObjectHashMap<ObjectIntHashMap<String>> categoryToIdMaps = new IntObjectHashMap<>();
        IntObjectHashMap<double[]> featureIdToBinEdges = new IntObjectHashMap<>();
        BitSet isCategorical = new BitSet();

        FeatureMeta[] featureMetas = state.statics.featureMetas;
        for (int k = 0; k < featureMetas.length; k += 1) {
            FeatureMeta featureMeta = featureMetas[k];
            if (featureMeta instanceof FeatureMeta.CategoricalFeatureMeta) {
                String[] categories = ((FeatureMeta.CategoricalFeatureMeta) featureMeta).categories;
                ObjectIntHashMap<String> categoryToId = new ObjectIntHashMap<>();
                for (int i = 0; i < categories.length; i += 1) {
                    categoryToId.put(categories[i], i);
                }
                categoryToIdMaps.put(k, categoryToId);
                isCategorical.set(k);
            } else if (featureMeta instanceof FeatureMeta.ContinuousFeatureMeta) {
                featureIdToBinEdges.put(
                        k, ((FeatureMeta.ContinuousFeatureMeta) featureMeta).binEdges);
            }
        }
        return new GBTModelData(
                state.statics.params.taskType.name(),
                state.statics.params.isInputVector,
                state.statics.prior,
                state.statics.params.stepSize,
                state.dynamics.roots,
                categoryToIdMaps,
                featureIdToBinEdges,
                isCategorical);
    }

    public static DataStream<GBTModelData> getModelDataStream(Table modelDataTable) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        return tEnv.toDataStream(modelDataTable).map(x -> x.getFieldAs(0));
    }

    /** The mapping computation is from {@link StringIndexerModel}. */
    private static int mapCategoricalFeature(ObjectIntHashMap<String> categoryToId, Object v) {
        String s;
        if (v instanceof String) {
            s = (String) v;
        } else if (v instanceof Number) {
            s = String.valueOf(v);
        } else if (null == v) {
            s = null;
        } else {
            throw new RuntimeException("Categorical column only supports string and numeric type.");
        }
        return categoryToId.getIfAbsent(s, categoryToId.size());
    }

    public IntDoubleHashMap rowToFeatures(Row row, String[] featureCols, String vectorCol) {
        IntDoubleHashMap features = new IntDoubleHashMap();
        if (isInputVector) {
            Vector vec = row.getFieldAs(vectorCol);
            SparseVector sv = vec.toSparse();
            for (int i = 0; i < sv.indices.length; i += 1) {
                features.put(sv.indices[i], sv.values[i]);
            }
        } else {
            for (int i = 0; i < featureCols.length; i += 1) {
                Object obj = row.getField(featureCols[i]);
                double v;
                if (isCategorical.get(i)) {
                    v = mapCategoricalFeature(categoryToIdMaps.get(i), obj);
                } else {
                    Number number = (Number) obj;
                    v = (null == number) ? Double.NaN : number.doubleValue();
                }
                features.put(i, v);
            }
        }
        return features;
    }

    public double predictRaw(IntDoubleHashMap rawFeatures) {
        double v = prior;
        for (Node root : roots) {
            Node node = root;
            while (!node.isLeaf) {
                boolean goLeft = node.split.shouldGoLeft(rawFeatures);
                node = goLeft ? node.left : node.right;
            }
            v += stepSize * node.split.prediction;
        }
        return v;
    }

    @Override
    public String toString() {
        return String.format(
                "GBTModelData{type=%s, prior=%s, roots=%s, categoryToIdMaps=%s, featureIdToBinEdges=%s, isCategorical=%s}",
                type, prior, roots, categoryToIdMaps, featureIdToBinEdges, isCategorical);
    }

    /** Encoder for {@link GBTModelData}. */
    public static class ModelDataEncoder implements Encoder<GBTModelData> {
        @Override
        public void encode(GBTModelData modelData, OutputStream outputStream) throws IOException {
            DataOutputView dataOutputView = new DataOutputViewStreamWrapper(outputStream);
            final GBTModelDataSerializer serializer = GBTModelDataSerializer.INSTANCE;
            serializer.serialize(modelData, dataOutputView);
        }
    }

    /** Decoder for {@link GBTModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<GBTModelData> {
        @Override
        public Reader<GBTModelData> createReader(Configuration config, FSDataInputStream stream) {
            return new Reader<GBTModelData>() {

                private final GBTModelDataSerializer serializer = GBTModelDataSerializer.INSTANCE;

                @Override
                public GBTModelData read() {
                    DataInputView source = new DataInputViewStreamWrapper(stream);
                    try {
                        return serializer.deserialize(source);
                    } catch (IOException e) {
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
        public TypeInformation<GBTModelData> getProducedType() {
            return TypeInformation.of(GBTModelData.class);
        }
    }
}
