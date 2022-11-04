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

/** Class for the double parameter. */
public class DoubleParam extends Param<Double> {

    public DoubleParam(
            String name,
            String description,
            Double defaultValue,
            ParamValidator<Double> validator) {
        super(name, Double.class, description, defaultValue, validator);
    }

    public DoubleParam(String name, String description, Double defaultValue) {
        this(name, description, defaultValue, ParamValidators.alwaysTrue());
    }

    @Override
    public Double jsonDecode(Object json) throws IOException {
        if (json instanceof String) {
            return Double.valueOf((String) json);
        }
        return (Double) json;
    }
}
