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

package org.apache.flink.ml.servable.builder;

import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.api.ModelServable;
import org.apache.flink.ml.servable.api.Row;
import org.apache.flink.ml.servable.api.TransformerServable;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ServableReadWriteUtils;
import org.apache.flink.util.Preconditions;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Defines Servable subclasses to be used in unit tests. */
public class ExampleServables {

    /**
     * A {@link TransformerServable} subclass that increments every value in the input dataframe by
     * `delta` and outputs the resulting values.
     */
    public static class SumModelServable implements ModelServable<SumModelServable> {

        private static final String COL_NAME = "input";

        private final Map<Param<?>, Object> paramMap = new HashMap<>();

        private int delta;

        public SumModelServable() {
            ParamUtils.initializeMapWithDefaultValues(paramMap, this);
        }

        @Override
        public DataFrame transform(DataFrame input) {
            List<Row> outputRows = new ArrayList<>();
            for (Row row : input.collect()) {
                Preconditions.checkState(row.size() == 1);
                int originValue = (Integer) row.get(0);
                outputRows.add(new Row(Collections.singletonList(originValue + delta)));
            }
            return new DataFrame(
                    Collections.singletonList(COL_NAME),
                    Collections.singletonList(DataTypes.INT),
                    outputRows);
        }

        @Override
        public Map<Param<?>, Object> getParamMap() {
            return paramMap;
        }

        public static SumModelServable load(String path) throws IOException {
            SumModelServable servable =
                    ServableReadWriteUtils.loadServableParam(path, SumModelServable.class);

            try (InputStream inputStream = ServableReadWriteUtils.loadModelData(path)) {
                DataInputViewStreamWrapper dataInputViewStreamWrapper =
                        new DataInputViewStreamWrapper(inputStream);
                servable.delta = IntSerializer.INSTANCE.deserialize(dataInputViewStreamWrapper);
                return servable;
            }
        }

        public SumModelServable setModelData(InputStream... modelDataInputs) throws IOException {
            Preconditions.checkArgument(modelDataInputs.length == 1);

            DataInputViewStreamWrapper inputViewStreamWrapper =
                    new DataInputViewStreamWrapper(modelDataInputs[0]);

            delta = IntSerializer.INSTANCE.deserialize(inputViewStreamWrapper);

            return this;
        }

        public static byte[] serialize(Object modelData) throws IOException {
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

            DataOutputViewStreamWrapper outputViewStreamWrapper =
                    new DataOutputViewStreamWrapper(outputStream);

            IntSerializer.INSTANCE.serialize((Integer) modelData, outputViewStreamWrapper);

            return outputStream.toByteArray();
        }
    }
}
