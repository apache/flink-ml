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

package org.apache.flink.iteration.config;

import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.CoreOptions;

import static org.apache.flink.configuration.ConfigOptions.key;

/** The options for the iteration. */
public class IterationOptions {

    public static final ConfigOption<String> DATA_CACHE_PATH =
            key("iteration.data-cache.path")
                    .stringType()
                    .noDefaultValue()
                    .withDescription(
                            "The base path of the data cached used inside the iteration. "
                                    + "If not specified, it will use local path randomly chosen from "
                                    + CoreOptions.TMP_DIRS.key());
}
