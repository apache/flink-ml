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

package org.apache.flink.ml.param;

import org.apache.flink.ml.util.JsonUtils;

import java.io.IOException;

/** Class for the array parameter. */
public class ArrayArrayParam<T> extends Param<T[][]> {

    public ArrayArrayParam(
            String name,
            Class<T[][]> clazz,
            String description,
            T[][] defaultValue,
            ParamValidator<T[][]> validator) {
        super(name, clazz, description, defaultValue, validator);
    }

    @Override
    public T[][] jsonDecode(Object json) throws IOException {
        String jsonStr = JsonUtils.OBJECT_MAPPER.writeValueAsString(json);
        return JsonUtils.OBJECT_MAPPER.readValue(jsonStr, clazz);
    }
}
