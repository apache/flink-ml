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

import java.io.IOException;

/** Class for the float parameter. */
public class FloatParam extends Param<Float> {

    public FloatParam(
            String name, String description, Float defaultValue, ParamValidator<Float> validator) {
        super(name, Float.class, description, defaultValue, validator);
    }

    public FloatParam(String name, String description, Float defaultValue) {
        this(name, description, defaultValue, ParamValidators.alwaysTrue());
    }

    @Override
    public Float jsonDecode(Object json) throws IOException {
        if (json instanceof Double) {
            return ((Double) json).floatValue();
        } else if (json instanceof String) {
            return Float.valueOf((String) json);
        }
        return (Float) json;
    }
}
