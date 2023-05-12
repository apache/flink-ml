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

package org.apache.flink.ml.util;

import org.apache.flink.core.fs.FileStatus;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.fs.Path;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.servable.api.TransformerServable;
import org.apache.flink.ml.servable.builder.PipelineModelServable;
import org.apache.flink.util.InstantiationUtil;

import java.io.IOException;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.flink.ml.util.FileUtils.loadMetadata;

/** Utility methods for loading Servables. */
public class ServableReadWriteUtils {

    /**
     * Loads the servables of a {@link PipelineModelServable} from the given path.
     *
     * <p>The method throws RuntimeException if the expectedClassName is not empty AND it does not
     * match the className of the previously saved PipelineModel.
     *
     * @param path The parent directory to load the PipelineModelServable metadata and its
     *     servables.
     * @return A list of servables.
     */
    public static List<TransformerServable<?>> loadPipeline(String path) throws IOException {
        Map<String, ?> metadata = loadMetadata(path, "");
        int numStages = (Integer) metadata.get("numStages");
        List<TransformerServable<?>> servables = new ArrayList<>(numStages);

        for (int i = 0; i < numStages; i++) {
            String stagePath = FileUtils.getPathForPipelineStage(i, numStages, path);
            servables.add(loadServable(stagePath));
        }
        return servables;
    }

    /**
     * Loads the {@link TransformerServable} from the given path by invoking the static
     * loadServable() method of the stage. The stage class name is read from the metadata file under
     * the given path. The loadServable() method is expected to construct the TransformerServable
     * instance with the saved parameters, model data and other metadata if exists.
     *
     * <p>Required: the stage class must have a static loadServable() method.
     *
     * @param path The parent directory of the stage metadata file.
     * @return An instance of {@link TransformerServable}.
     */
    private static TransformerServable<?> loadServable(String path) throws IOException {
        Map<String, ?> metadata = FileUtils.loadMetadata(path, "");
        String className = (String) metadata.get("className");

        try {
            Class<?> clazz = Class.forName(className);
            Method method = clazz.getMethod("loadServable", String.class);
            method.setAccessible(true);
            return (TransformerServable<?>) method.invoke(null, path);
        } catch (NoSuchMethodException e) {
            String methodName = String.format("%s::loadServable(String)", className);
            throw new RuntimeException(
                    "Failed to load servable because the static method "
                            + methodName
                            + " is not implemented.",
                    e);
        } catch (ClassNotFoundException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException("Failed to load servable.", e);
        }
    }

    /**
     * Loads the {@link TransformerServable} with the saved parameters from the given path. This
     * method reads the metadata file under the given path, instantiates the servable using its
     * no-argument constructor, and loads the servable with the paramMap from the metadata file.
     *
     * <p>Note: This method does not attempt to read model data from the given path. Caller needs to
     * read and deserialize model data from the given path.
     *
     * <p>Required: the class with type T must have a no-argument constructor.
     *
     * @param path The parent directory of the metadata file.
     * @param <T> The class type of the TransformerServable subclass.
     * @return An instance of class type T.
     */
    public static <T extends TransformerServable<T>> T loadServableParam(
            String path, Class<T> clazz) throws IOException {
        T instance = InstantiationUtil.instantiate(clazz);

        Map<String, Param<?>> nameToParam = new HashMap<>();
        for (Param<?> param : ParamUtils.getPublicFinalParamFields(instance)) {
            nameToParam.put(param.name, param);
        }

        Map<String, ?> jsonMap = loadMetadata(path, "");
        if (jsonMap.containsKey("paramMap")) {
            Map<String, Object> paramMap = (Map<String, Object>) jsonMap.get("paramMap");
            for (Map.Entry<String, Object> entry : paramMap.entrySet()) {
                Param<?> param = nameToParam.get(entry.getKey());
                ParamUtils.setParam(instance, param, param.jsonDecode(entry.getValue()));
            }
        }

        return instance;
    }

    /**
     * Opens an FSDataInputStream to read the model data file in the directory. Only one model data
     * file is expected to be in the directory.
     *
     * @param path The parent directory of the model data file.
     * @return A FSDataInputStream to read the model data.
     */
    public static InputStream loadModelData(String path) throws IOException {
        Path modelDataPath = FileUtils.getDataPath(path);

        FileSystem fileSystem = modelDataPath.getFileSystem();

        FileStatus[] files = fileSystem.listStatus(modelDataPath);
        List<InputStream> inputStreams = new ArrayList<>();
        for (FileStatus file : files) {
            inputStreams.add(fileSystem.open(file.getPath()));
        }
        return new SequenceInputStream(Collections.enumeration(inputStreams));
    }
}
