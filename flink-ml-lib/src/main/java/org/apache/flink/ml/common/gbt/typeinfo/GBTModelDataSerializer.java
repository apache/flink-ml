/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.flink.ml.common.gbt.typeinfo;

import org.apache.flink.api.common.typeutils.SimpleTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.base.BooleanSerializer;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.StringSerializer;
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.api.common.typeutils.base.array.BytePrimitiveArraySerializer;
import org.apache.flink.api.common.typeutils.base.array.DoublePrimitiveArraySerializer;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.gbt.GBTModelData;
import org.apache.flink.ml.common.gbt.defs.Node;

import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectIntHashMap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/** Specialized serializer for {@link GBTModelData}. */
public final class GBTModelDataSerializer extends TypeSerializerSingleton<GBTModelData> {

    public static final GBTModelDataSerializer INSTANCE = new GBTModelDataSerializer();
    private static final long serialVersionUID = 1L;
    private static final NodeSerializer NODE_SERIALIZER = NodeSerializer.INSTANCE;

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public GBTModelData createInstance() {
        return new GBTModelData();
    }

    @Override
    public GBTModelData copy(GBTModelData from) {
        GBTModelData record = new GBTModelData();
        record.type = from.type;
        record.isInputVector = from.isInputVector;

        record.prior = from.prior;
        record.stepSize = from.stepSize;

        record.allTrees = new ArrayList<>(from.allTrees.size());
        for (int i = 0; i < from.allTrees.size(); i += 1) {
            record.allTrees.add(new ArrayList<>(from.allTrees.get(i)));
        }
        record.featureNames = new ArrayList<>(from.featureNames);
        record.categoryToIdMaps = new IntObjectHashMap<>(from.categoryToIdMaps);
        record.featureIdToBinEdges = new IntObjectHashMap<>(from.featureIdToBinEdges);
        record.isCategorical = BitSet.valueOf(from.isCategorical.toByteArray());
        return record;
    }

    @Override
    public GBTModelData copy(GBTModelData from, GBTModelData reuse) {
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(GBTModelData record, DataOutputView target) throws IOException {
        StringSerializer.INSTANCE.serialize(record.type, target);
        BooleanSerializer.INSTANCE.serialize(record.isInputVector, target);

        DoubleSerializer.INSTANCE.serialize(record.prior, target);
        DoubleSerializer.INSTANCE.serialize(record.stepSize, target);

        IntSerializer.INSTANCE.serialize(record.allTrees.size(), target);
        for (List<Node> treeNodes : record.allTrees) {
            IntSerializer.INSTANCE.serialize(treeNodes.size(), target);
            for (Node treeNode : treeNodes) {
                NodeSerializer.INSTANCE.serialize(treeNode, target);
            }
        }

        IntSerializer.INSTANCE.serialize(record.featureNames.size(), target);
        for (int i = 0; i < record.featureNames.size(); i += 1) {
            StringSerializer.INSTANCE.serialize(record.featureNames.get(i), target);
        }

        IntSerializer.INSTANCE.serialize(record.categoryToIdMaps.size(), target);
        for (int featureId : record.categoryToIdMaps.keysView().toArray()) {
            ObjectIntHashMap<String> categoryToIdMap = record.categoryToIdMaps.get(featureId);
            IntSerializer.INSTANCE.serialize(featureId, target);
            IntSerializer.INSTANCE.serialize(categoryToIdMap.size(), target);
            for (String category : categoryToIdMap.keysView()) {
                StringSerializer.INSTANCE.serialize(category, target);
                IntSerializer.INSTANCE.serialize(categoryToIdMap.get(category), target);
            }
        }

        IntSerializer.INSTANCE.serialize(record.featureIdToBinEdges.size(), target);
        for (int featureId : record.featureIdToBinEdges.keysView().toArray()) {
            double[] binEdges = record.featureIdToBinEdges.get(featureId);
            IntSerializer.INSTANCE.serialize(featureId, target);
            DoublePrimitiveArraySerializer.INSTANCE.serialize(binEdges, target);
        }

        BytePrimitiveArraySerializer.INSTANCE.serialize(record.isCategorical.toByteArray(), target);
    }

    @Override
    public GBTModelData deserialize(DataInputView source) throws IOException {
        GBTModelData record = new GBTModelData();

        record.type = StringSerializer.INSTANCE.deserialize(source);
        record.isInputVector = BooleanSerializer.INSTANCE.deserialize(source);

        record.prior = DoubleSerializer.INSTANCE.deserialize(source);
        record.stepSize = DoubleSerializer.INSTANCE.deserialize(source);

        int numTrees = IntSerializer.INSTANCE.deserialize(source);
        record.allTrees = new ArrayList<>(numTrees);
        for (int k = 0; k < numTrees; k += 1) {
            int numTreeNodes = IntSerializer.INSTANCE.deserialize(source);
            List<Node> treeNodes = new ArrayList<>(numTreeNodes);
            for (int i = 0; i < numTreeNodes; i += 1) {
                treeNodes.add(NODE_SERIALIZER.deserialize(source));
            }
            record.allTrees.add(treeNodes);
        }

        int numFeatures = IntSerializer.INSTANCE.deserialize(source);
        record.featureNames = new ArrayList<>(numFeatures);
        for (int k = 0; k < numFeatures; k += 1) {
            String featureName = StringSerializer.INSTANCE.deserialize(source);
            record.featureNames.add(featureName);
        }

        int numCategoricalFeatures = IntSerializer.INSTANCE.deserialize(source);
        record.categoryToIdMaps = IntObjectHashMap.newMap();
        for (int k = 0; k < numCategoricalFeatures; k += 1) {
            int featureId = IntSerializer.INSTANCE.deserialize(source);
            int categoryToIdMapSize = IntSerializer.INSTANCE.deserialize(source);
            ObjectIntHashMap<String> categoryToIdMap = ObjectIntHashMap.newMap();
            for (int i = 0; i < categoryToIdMapSize; i += 1) {
                categoryToIdMap.put(
                        StringSerializer.INSTANCE.deserialize(source),
                        IntSerializer.INSTANCE.deserialize(source));
            }
            record.categoryToIdMaps.put(featureId, categoryToIdMap);
        }

        int numContinuousFeatures = IntSerializer.INSTANCE.deserialize(source);
        record.featureIdToBinEdges = IntObjectHashMap.newMap();
        for (int i = 0; i < numContinuousFeatures; i += 1) {
            int featureId = IntSerializer.INSTANCE.deserialize(source);
            double[] binEdges = DoublePrimitiveArraySerializer.INSTANCE.deserialize(source);
            record.featureIdToBinEdges.put(featureId, binEdges);
        }

        record.isCategorical =
                BitSet.valueOf(BytePrimitiveArraySerializer.INSTANCE.deserialize(source));
        return record;
    }

    @Override
    public GBTModelData deserialize(GBTModelData reuse, DataInputView source) throws IOException {
        return deserialize(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<GBTModelData> snapshotConfiguration() {
        return new GBTModelDataSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class GBTModelDataSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<GBTModelData> {

        public GBTModelDataSerializerSnapshot() {
            super(GBTModelDataSerializer::new);
        }
    }
}
