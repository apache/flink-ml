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

package org.apache.flink.iteration.typeinfo;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.iteration.IterationRecord;

import java.util.Objects;

/** The type information for {@link IterationRecord}. */
public class IterationRecordTypeInfo<T> extends TypeInformation<IterationRecord<T>> {

    private final TypeInformation<T> innerTypeInfo;

    public IterationRecordTypeInfo(TypeInformation<T> innerTypeInfo) {
        this.innerTypeInfo = innerTypeInfo;
    }

    public TypeInformation<T> getInnerTypeInfo() {
        return innerTypeInfo;
    }

    public boolean isBasicType() {
        return false;
    }

    @Override
    public boolean isTupleType() {
        return false;
    }

    @Override
    public int getArity() {
        return 1;
    }

    @Override
    public int getTotalFields() {
        return 1;
    }

    @Override
    public Class<IterationRecord<T>> getTypeClass() {
        return (Class) IterationRecord.class;
    }

    @Override
    public boolean isKeyType() {
        return false;
    }

    @Override
    public TypeSerializer<IterationRecord<T>> createSerializer(ExecutionConfig config) {
        return new IterationRecordSerializer<>(innerTypeInfo.createSerializer(config));
    }

    @Override
    public String toString() {
        return "IterationRecord<" + innerTypeInfo + ">";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }

        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        IterationRecordTypeInfo<T> that = (IterationRecordTypeInfo<T>) o;
        return Objects.equals(innerTypeInfo, that.innerTypeInfo);
    }

    @Override
    public int hashCode() {
        return innerTypeInfo != null ? innerTypeInfo.hashCode() : 0;
    }

    @Override
    public boolean canEqual(Object obj) {
        return obj instanceof IterationRecordTypeInfo;
    }
}
