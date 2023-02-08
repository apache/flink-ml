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

package org.apache.flink.ml.common.typeinfo;

import org.apache.flink.api.common.typeinfo.TypeInfoFactory;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.TypeExtractor;
import org.apache.flink.ml.common.util.QuantileSummary;

import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Map;

/**
 * Used by {@link TypeExtractor} to create a {@link TypeInformation} for implementations of {@link
 * QuantileSummary}.
 */
public class QuantileSummaryTypeInfoFactory extends TypeInfoFactory<QuantileSummary> {

    private static final Map<String, TypeInformation<?>> fields;

    static {
        fields = new HashMap<>();
        fields.put("relativeError", Types.DOUBLE);
        fields.put("compressThreshold", Types.INT);
        fields.put("count", Types.LONG);
        fields.put("sampled", Types.LIST(TypeInformation.of(QuantileSummary.StatsTuple.class)));
        fields.put("headBuffer", Types.LIST(Types.DOUBLE));
        fields.put("compressed", Types.BOOLEAN);
    }

    private static final TypeInformation<QuantileSummary> TYPE_INFO =
            Types.POJO(QuantileSummary.class, fields);

    @Override
    public TypeInformation<QuantileSummary> createTypeInfo(
            Type t, Map<String, TypeInformation<?>> genericParameters) {
        return TYPE_INFO;
    }
}
