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

package org.apache.flink.ml.feature.lsh;

import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.IOException;

/** A Model which generates hash values using the model data computed by {@link MinHashLSH}. */
public class MinHashLSHModel extends LSHModel<MinHashLSHModel> {

    public MinHashLSHModel() {
        super(MinHashLSHModelData.class);
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();

        ReadWriteUtils.saveModelData(
                tEnv.toDataStream(modelDataTable, MinHashLSHModelData.class),
                path,
                new MinHashLSHModelData.ModelDataEncoder());
    }

    /**
     * Loads model data from path.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path Model path.
     * @return LSH model.
     */
    public static MinHashLSHModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        MinHashLSHModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new MinHashLSHModelData.ModelDataDecoder());
        model.setModelData(modelDataTable);
        return model;
    }
}
