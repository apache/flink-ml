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

package org.apache.flink.ml.common.param;

import org.apache.flink.ml.common.window.GlobalWindows;
import org.apache.flink.ml.common.window.Windows;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.WindowsParam;
import org.apache.flink.ml.param.WithParams;

/** Interface for the shared windows param. */
public interface HasWindows<T> extends WithParams<T> {
    Param<Windows> WINDOWS =
            new WindowsParam(
                    "windows",
                    "Windowing strategy that determines how to create mini-batches from input data.",
                    GlobalWindows.getInstance(),
                    ParamValidators.notNull());

    default Windows getWindows() {
        return get(WINDOWS);
    }

    default T setWindows(Windows windows) {
        return set(WINDOWS, windows);
    }
}
