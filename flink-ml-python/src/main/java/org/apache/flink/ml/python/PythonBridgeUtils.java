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

package org.apache.flink.ml.python;

import org.apache.flink.api.common.typeinfo.BasicArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.SqlTimeTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.tuple.Tuple;
import org.apache.flink.api.java.typeutils.ListTypeInfo;
import org.apache.flink.api.java.typeutils.MapTypeInfo;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.api.java.typeutils.TupleTypeInfoBase;
import org.apache.flink.api.python.shaded.net.razorvine.pickle.Pickler;
import org.apache.flink.core.memory.ByteArrayOutputStreamWithPos;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.streaming.api.typeinfo.python.PickledByteArrayTypeInfo;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.sql.Date;
import java.sql.Time;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.apache.flink.api.common.typeinfo.BasicTypeInfo.FLOAT_TYPE_INFO;
import static org.apache.flink.api.common.typeinfo.SqlTimeTypeInfo.DATE;
import static org.apache.flink.api.common.typeinfo.SqlTimeTypeInfo.TIME;

/**
 * Utility functions used to override PyFlink methods to provide a temporary solution for certain
 * bugs.
 */
// TODO: Remove this class after Flink ML depends on a Flink version with FLINK-30168 and
// FLINK-29477 fixed.
public class PythonBridgeUtils {
    public static Object getPickledBytesFromJavaObject(Object obj, TypeInformation<?> dataType)
            throws IOException {
        Pickler pickler = new Pickler();

        // triggers the initialization process
        org.apache.flink.api.common.python.PythonBridgeUtils.getPickledBytesFromJavaObject(
                null, null);

        if (obj == null) {
            return new byte[0];
        } else {
            if (dataType instanceof SqlTimeTypeInfo) {
                SqlTimeTypeInfo<?> sqlTimeTypeInfo =
                        SqlTimeTypeInfo.getInfoFor(dataType.getTypeClass());
                if (sqlTimeTypeInfo == DATE) {
                    return pickler.dumps(((Date) obj).toLocalDate().toEpochDay());
                } else if (sqlTimeTypeInfo == TIME) {
                    return pickler.dumps(((Time) obj).toLocalTime().toNanoOfDay() / 1000);
                }
            } else if (dataType instanceof RowTypeInfo || dataType instanceof TupleTypeInfo) {
                TypeInformation<?>[] fieldTypes = ((TupleTypeInfoBase<?>) dataType).getFieldTypes();
                int arity =
                        dataType instanceof RowTypeInfo
                                ? ((Row) obj).getArity()
                                : ((Tuple) obj).getArity();

                List<Object> fieldBytes = new ArrayList<>(arity + 1);
                if (dataType instanceof RowTypeInfo) {
                    fieldBytes.add(new byte[] {((Row) obj).getKind().toByteValue()});
                }
                for (int i = 0; i < arity; i++) {
                    Object field =
                            dataType instanceof RowTypeInfo
                                    ? ((Row) obj).getField(i)
                                    : ((Tuple) obj).getField(i);
                    fieldBytes.add(getPickledBytesFromJavaObject(field, fieldTypes[i]));
                }
                return fieldBytes;
            } else if (dataType instanceof BasicArrayTypeInfo
                    || dataType instanceof PrimitiveArrayTypeInfo
                    || dataType instanceof ObjectArrayTypeInfo) {
                Object[] objects;
                TypeInformation<?> elementType;
                if (dataType instanceof BasicArrayTypeInfo) {
                    objects = (Object[]) obj;
                    elementType = ((BasicArrayTypeInfo<?, ?>) dataType).getComponentInfo();
                } else if (dataType instanceof PrimitiveArrayTypeInfo) {
                    objects = primitiveArrayConverter(obj, dataType);
                    elementType = ((PrimitiveArrayTypeInfo<?>) dataType).getComponentType();
                } else {
                    objects = (Object[]) obj;
                    elementType = ((ObjectArrayTypeInfo<?, ?>) dataType).getComponentInfo();
                }

                List<Object> serializedElements = new ArrayList<>(objects.length);

                for (Object object : objects) {
                    serializedElements.add(getPickledBytesFromJavaObject(object, elementType));
                }
                return pickler.dumps(serializedElements);
            } else if (dataType instanceof MapTypeInfo) {
                List<List<Object>> serializedMapKV = new ArrayList<>(2);
                Map<Object, Object> mapObj = (Map) obj;
                List<Object> keyBytesList = new ArrayList<>(mapObj.size());
                List<Object> valueBytesList = new ArrayList<>(mapObj.size());
                for (Map.Entry entry : mapObj.entrySet()) {
                    keyBytesList.add(
                            getPickledBytesFromJavaObject(
                                    entry.getKey(), ((MapTypeInfo) dataType).getKeyTypeInfo()));
                    valueBytesList.add(
                            getPickledBytesFromJavaObject(
                                    entry.getValue(), ((MapTypeInfo) dataType).getValueTypeInfo()));
                }
                serializedMapKV.add(keyBytesList);
                serializedMapKV.add(valueBytesList);
                return pickler.dumps(serializedMapKV);
            } else if (dataType instanceof ListTypeInfo) {
                List objects = (List) obj;
                List<Object> serializedElements = new ArrayList<>(objects.size());
                TypeInformation elementType = ((ListTypeInfo) dataType).getElementTypeInfo();
                for (Object object : objects) {
                    serializedElements.add(getPickledBytesFromJavaObject(object, elementType));
                }
                return pickler.dumps(serializedElements);
            }
            if (dataType instanceof BasicTypeInfo
                    && BasicTypeInfo.getInfoFor(dataType.getTypeClass()) == FLOAT_TYPE_INFO) {
                // Serialization of float type with pickler loses precision.
                return pickler.dumps(String.valueOf(obj));
            } else if (dataType instanceof PickledByteArrayTypeInfo
                    || dataType instanceof BasicTypeInfo) {
                return pickler.dumps(obj);
            } else {
                // other typeinfos will use the corresponding serializer to serialize data.
                TypeSerializer serializer = dataType.createSerializer(null);
                ByteArrayOutputStreamWithPos baos = new ByteArrayOutputStreamWithPos();
                DataOutputViewStreamWrapper baosWrapper = new DataOutputViewStreamWrapper(baos);
                serializer.serialize(obj, baosWrapper);
                return pickler.dumps(baos.toByteArray());
            }
        }
    }

    private static Object[] primitiveArrayConverter(
            Object array, TypeInformation<?> arrayTypeInfo) {
        Preconditions.checkArgument(arrayTypeInfo instanceof PrimitiveArrayTypeInfo);
        Preconditions.checkArgument(array.getClass().isArray());
        Object[] objects;
        if (PrimitiveArrayTypeInfo.BOOLEAN_PRIMITIVE_ARRAY_TYPE_INFO.equals(arrayTypeInfo)) {
            boolean[] booleans = (boolean[]) array;
            objects = new Object[booleans.length];
            for (int i = 0; i < booleans.length; i++) {
                objects[i] = booleans[i];
            }
        } else if (PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO.equals(arrayTypeInfo)) {
            byte[] bytes = (byte[]) array;
            objects = new Object[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                objects[i] = bytes[i];
            }
        } else if (PrimitiveArrayTypeInfo.SHORT_PRIMITIVE_ARRAY_TYPE_INFO.equals(arrayTypeInfo)) {
            short[] shorts = (short[]) array;
            objects = new Object[shorts.length];
            for (int i = 0; i < shorts.length; i++) {
                objects[i] = shorts[i];
            }
        } else if (PrimitiveArrayTypeInfo.INT_PRIMITIVE_ARRAY_TYPE_INFO.equals(arrayTypeInfo)) {
            int[] ints = (int[]) array;
            objects = new Object[ints.length];
            for (int i = 0; i < ints.length; i++) {
                objects[i] = ints[i];
            }
        } else if (PrimitiveArrayTypeInfo.LONG_PRIMITIVE_ARRAY_TYPE_INFO.equals(arrayTypeInfo)) {
            long[] longs = (long[]) array;
            objects = new Object[longs.length];
            for (int i = 0; i < longs.length; i++) {
                objects[i] = longs[i];
            }
        } else if (PrimitiveArrayTypeInfo.FLOAT_PRIMITIVE_ARRAY_TYPE_INFO.equals(arrayTypeInfo)) {
            float[] floats = (float[]) array;
            objects = new Object[floats.length];
            for (int i = 0; i < floats.length; i++) {
                objects[i] = floats[i];
            }
        } else if (PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO.equals(arrayTypeInfo)) {
            double[] doubles = (double[]) array;
            objects = new Object[doubles.length];
            for (int i = 0; i < doubles.length; i++) {
                objects[i] = doubles[i];
            }
        } else if (PrimitiveArrayTypeInfo.CHAR_PRIMITIVE_ARRAY_TYPE_INFO.equals(arrayTypeInfo)) {
            char[] chars = (char[]) array;
            objects = new Object[chars.length];
            for (int i = 0; i < chars.length; i++) {
                objects[i] = chars[i];
            }
        } else {
            throw new UnsupportedOperationException(
                    String.format(
                            "Primitive array of %s is not supported in PyFlink yet",
                            ((PrimitiveArrayTypeInfo<?>) arrayTypeInfo)
                                    .getComponentType()
                                    .getTypeClass()
                                    .getSimpleName()));
        }
        return objects;
    }
}
