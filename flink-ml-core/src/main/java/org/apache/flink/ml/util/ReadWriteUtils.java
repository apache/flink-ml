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

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.connector.source.Source;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.fs.Path;
import org.apache.flink.ml.api.Stage;
import org.apache.flink.ml.builder.Graph;
import org.apache.flink.ml.builder.GraphData;
import org.apache.flink.ml.builder.GraphModel;
import org.apache.flink.ml.builder.GraphNode;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.WithParams;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.filesystem.bucketassigners.BasePathBucketAssigner;
import org.apache.flink.streaming.api.functions.sink.filesystem.rollingpolicies.OnCheckpointRollingPolicy;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.InstantiationUtil;
import org.apache.flink.util.Preconditions;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.JsonParser;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Utility methods for reading and writing stages. */
public class ReadWriteUtils {
    public static final ObjectMapper OBJECT_MAPPER =
            new ObjectMapper().enable(JsonParser.Feature.ALLOW_COMMENTS);

    // A helper method that calls encodes the given parameter value to a json string. We can not
    // call param.jsonEncode(value) directly because Param::jsonEncode(...) needs the actual type
    // of the value.
    private static <T> Object jsonEncodeHelper(Param<T> param, Object value) throws IOException {
        return param.jsonEncode((T) value);
    }

    // Converts Map<Param<?>, Object> to Map<String, Object> which maps the parameter name to the
    // json-supported parameter value.
    private static Map<String, Object> jsonEncode(Map<Param<?>, Object> paramMap)
            throws IOException {
        Map<String, Object> result = new HashMap<>(paramMap.size());
        for (Map.Entry<Param<?>, Object> entry : paramMap.entrySet()) {
            Object json = jsonEncodeHelper(entry.getKey(), entry.getValue());
            result.put(entry.getKey().name, json);
        }
        return result;
    }

    /**
     * Saves the metadata of the given stage and the extra metadata to a file named `metadata` under
     * the given path. The metadata of a stage includes the stage class name, parameter values etc.
     *
     * <p>Required: the metadata file under the given path should not exist.
     *
     * @param stage The stage instance.
     * @param path The parent directory to save the stage metadata.
     * @param extraMetadata The extra metadata to be saved.
     */
    public static void saveMetadata(Stage<?> stage, String path, Map<String, ?> extraMetadata)
            throws IOException {
        Map<String, Object> metadata = new HashMap<>(extraMetadata);
        metadata.put("className", stage.getClass().getName());
        metadata.put("timestamp", System.currentTimeMillis());
        metadata.put("paramMap", jsonEncode(stage.getParamMap()));
        // TODO: add version in the metadata.
        String metadataStr = OBJECT_MAPPER.writeValueAsString(metadata);

        saveToFile(new Path(path, "metadata").toUri().toString(), metadataStr, false);
    }

    /** Saves a given string to the specified file. */
    public static void saveToFile(String pathStr, String content, boolean isOverwrite)
            throws IOException {
        Path path = new Path(pathStr);

        // Creates parent directories if not already created.
        FileSystem fs = mkdirs(path.getParent());

        FileSystem.WriteMode writeMode = FileSystem.WriteMode.OVERWRITE;
        if (!isOverwrite) {
            writeMode = FileSystem.WriteMode.NO_OVERWRITE;
            if (fs.exists(path)) {
                throw new IOException("File " + path + " already exists.");
            }
        }
        try (BufferedWriter writer =
                new BufferedWriter(new OutputStreamWriter(fs.create(path, writeMode)))) {
            writer.write(content);
        }
    }

    /**
     * Saves the metadata of the given stage to a file named `metadata` under the given path. The
     * metadata of a stage includes the stage class name, parameter values etc.
     *
     * <p>Required: the metadata file under the given path should not exist.
     *
     * @param stage The stage instance.
     * @param path The parent directory to save the stage metadata.
     */
    public static void saveMetadata(Stage<?> stage, String path) throws IOException {
        saveMetadata(stage, path, new HashMap<>());
    }

    /** Returns a subdirectory of the given path for saving/loading model data. */
    private static String getDataPath(String path) {
        return new Path(path, "data").toString();
    }

    /**
     * Loads the metadata from the metadata file under the given path.
     *
     * <p>The method throws RuntimeException if the expectedClassName is not empty AND it does not
     * match the className of the previously saved stage.
     *
     * @param path The parent directory of the metadata file to read from.
     * @param expectedClassName The expected class name of the stage.
     * @return A map from metadata name to metadata value.
     */
    public static Map<String, ?> loadMetadata(String path, String expectedClassName)
            throws IOException {
        Path metadataPath = new Path(path, "metadata");
        FileSystem fs = metadataPath.getFileSystem();

        StringBuilder buffer = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(metadataPath)))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.startsWith("#")) {
                    buffer.append(line);
                }
            }
        }

        @SuppressWarnings("unchecked")
        Map<String, ?> result = OBJECT_MAPPER.readValue(buffer.toString(), Map.class);

        String className = (String) result.get("className");
        if (!expectedClassName.isEmpty() && !expectedClassName.equals(className)) {
            throw new RuntimeException(
                    "Class name "
                            + className
                            + " does not match the expected class name "
                            + expectedClassName
                            + ".");
        }

        return result;
    }

    // Returns a string with value {parentPath}/stages/{stageIdx}, where the stageIdx is prefixed
    // with zero or more `0` to have the same length as numStages. The resulting string can be
    // used as the directory to save a stage of the Pipeline or PipelineModel.
    private static String getPathForPipelineStage(int stageIdx, int numStages, String parentPath) {
        String format =
                String.format("stages%s%%0%dd", File.separator, String.valueOf(numStages).length());
        String fileName = String.format(format, stageIdx);
        return new Path(parentPath, fileName).toString();
    }

    /**
     * Saves a Pipeline or PipelineModel with the given list of stages to the given path.
     *
     * @param pipeline A Pipeline or PipelineModel instance.
     * @param stages A list of stages of the given pipeline.
     * @param path The parent directory to save the pipeline metadata and its stages.
     */
    public static void savePipeline(Stage<?> pipeline, List<Stage<?>> stages, String path)
            throws IOException {
        // Creates parent directories if not already created.
        mkdirs(new Path(path));

        Map<String, Object> extraMetadata = new HashMap<>();
        extraMetadata.put("numStages", stages.size());
        saveMetadata(pipeline, path, extraMetadata);

        int numStages = stages.size();
        for (int i = 0; i < numStages; i++) {
            String stagePath = getPathForPipelineStage(i, numStages, path);
            stages.get(i).save(stagePath);
        }
    }

    /**
     * Loads the stages of a Pipeline or PipelineModel from the given path.
     *
     * <p>The method throws RuntimeException if the expectedClassName is not empty AND it does not
     * match the className of the previously saved Pipeline or PipelineModel.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path The parent directory to load the pipeline metadata and its stages.
     * @param expectedClassName The expected class name of the pipeline.
     * @return A list of stages.
     */
    public static List<Stage<?>> loadPipeline(
            StreamTableEnvironment tEnv, String path, String expectedClassName) throws IOException {
        Map<String, ?> metadata = loadMetadata(path, expectedClassName);
        int numStages = (Integer) metadata.get("numStages");
        List<Stage<?>> stages = new ArrayList<>(numStages);

        for (int i = 0; i < numStages; i++) {
            String stagePath = getPathForPipelineStage(i, numStages, path);
            stages.add(loadStage(tEnv, stagePath));
        }
        return stages;
    }

    private static FileSystem mkdirs(Path path) throws IOException {
        FileSystem fs = path.getFileSystem();
        fs.mkdirs(path);
        return fs;
    }

    /**
     * Saves a Graph or GraphModel with the given GraphData to the given path.
     *
     * @param graph A Graph or GraphModel instance.
     * @param graphData A GraphData instance.
     * @param path The parent directory to save the graph metadata and its stages.
     */
    public static void saveGraph(Stage<?> graph, GraphData graphData, String path)
            throws IOException {
        // Creates parent directories if not already created.
        mkdirs(new Path(path));

        Map<String, Object> extraMetadata = new HashMap<>();
        extraMetadata.put("graphData", graphData.toMap());
        saveMetadata(graph, path, extraMetadata);
        int maxNodeId =
                graphData.nodes.stream()
                        .map(node -> node.nodeId)
                        .max(Comparator.naturalOrder())
                        .orElse(-1);

        for (GraphNode node : graphData.nodes) {
            String stagePath = getPathForPipelineStage(node.nodeId, maxNodeId + 1, path);
            node.stage.save(stagePath);
        }
    }

    /**
     * Loads a Graph or GraphModel from the given path.
     *
     * <p>The method throws RuntimeException if the expectedClassName is not empty AND it does not
     * match the className of the previously saved Pipeline or PipelineModel.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path The parent directory to load the pipeline metadata and its stages.
     * @param expectedClassName The expected class name of the pipeline.
     * @return A Graph or GraphModel instance.
     */
    @SuppressWarnings("unchecked")
    public static Stage<?> loadGraph(
            StreamTableEnvironment tEnv, String path, String expectedClassName) throws IOException {
        Map<String, ?> metadata = loadMetadata(path, expectedClassName);
        GraphData graphData = GraphData.fromMap((Map<String, Object>) metadata.get("graphData"));

        int maxNodeId =
                graphData.nodes.stream()
                        .map(node -> node.nodeId)
                        .max(Comparator.naturalOrder())
                        .orElse(-1);

        for (GraphNode node : graphData.nodes) {
            String stagePath = getPathForPipelineStage(node.nodeId, maxNodeId + 1, path);
            node.stage = loadStage(tEnv, stagePath);
        }

        if (expectedClassName.equals(GraphModel.class.getName())) {
            return new GraphModel(
                    graphData.nodes,
                    graphData.modelInputIds,
                    graphData.outputIds,
                    graphData.inputModelDataIds,
                    graphData.outputModelDataIds);
        }
        Preconditions.checkState(expectedClassName.equals(Graph.class.getName()));
        return new Graph(
                graphData.nodes,
                graphData.estimatorInputIds,
                graphData.modelInputIds,
                graphData.outputIds,
                graphData.inputModelDataIds,
                graphData.outputModelDataIds);
    }

    // A helper method that sets WithParams object's parameter value. We can not call
    // WithParams.set(param, value)
    // directly because WithParams::set(...) needs the actual type of the value.
    @SuppressWarnings("unchecked")
    public static <T> void setParam(WithParams<?> instance, Param<T> param, Object value) {
        instance.set(param, (T) value);
    }

    // A helper method that updates WithParams instance's param map using values from the
    // paramOverrides. This method only updates values for parameters already defined in the
    // instance's param map.
    public static void updateExistingParams(
            WithParams<?> instance, Map<Param<?>, Object> paramOverrides) {
        Set<Param<?>> existingParams = instance.getParamMap().keySet();
        for (Map.Entry<Param<?>, Object> entry : paramOverrides.entrySet()) {
            if (existingParams.contains(entry.getKey())) {
                setParam(instance, entry.getKey(), entry.getValue());
            }
        }
    }

    /**
     * Loads the stage with the saved parameters from the given path. This method reads the metadata
     * file under the given path, instantiates the stage using its no-argument constructor, and
     * loads the stage with the paramMap from the metadata file.
     *
     * <p>Note: This method does not attempt to read model data from the given path. Caller needs to
     * read model data from the given path if the stage has model data.
     *
     * <p>Required: the class with type T must have a no-argument constructor.
     *
     * @param path The parent directory of the stage metadata file.
     * @param <T> The class type of the Stage subclass.
     * @return An instance of class type T.
     */
    public static <T extends Stage<T>> T loadStageParam(String path) throws IOException {
        try {
            return instantiateWithParams(loadMetadata(path, ""));
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Failed to load stage.", e);
        }
    }

    /**
     * Instantiates a WithParams subclass from the provided json map.
     *
     * @param jsonMap a map containing className and paramMap.
     * @return the instantiated WithParams subclass instance.
     */
    @SuppressWarnings("unchecked")
    public static <T extends WithParams<T>> T instantiateWithParams(Map<String, ?> jsonMap)
            throws ClassNotFoundException, IOException {
        String className = (String) jsonMap.get("className");
        Class<T> clazz = (Class<T>) Class.forName(className);
        T instance = InstantiationUtil.instantiate(clazz);

        Map<String, Param<?>> nameToParam = new HashMap<>();
        for (Param<?> param : ParamUtils.getPublicFinalParamFields(instance)) {
            nameToParam.put(param.name, param);
        }

        if (jsonMap.containsKey("paramMap")) {
            Map<String, Object> paramMap = (Map<String, Object>) jsonMap.get("paramMap");
            for (Map.Entry<String, Object> entry : paramMap.entrySet()) {
                Param<?> param = nameToParam.get(entry.getKey());
                setParam(instance, param, param.jsonDecode(entry.getValue()));
            }
        }

        return instance;
    }

    /**
     * Loads the stage from the given path by invoking the static load() method of the stage. The
     * stage class name is read from the metadata file under the given path. The load() method is
     * expected to construct the stage instance with the saved parameters, model data and other
     * metadata if exists.
     *
     * <p>Required: the stage class must have a static load() method.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path The parent directory of the stage metadata file.
     * @return An instance of Stage.
     */
    public static Stage<?> loadStage(StreamTableEnvironment tEnv, String path) throws IOException {
        Map<String, ?> metadata = loadMetadata(path, "");
        String className = (String) metadata.get("className");

        try {
            Class<?> clazz = Class.forName(className);
            Method method = clazz.getMethod("load", StreamTableEnvironment.class, String.class);
            method.setAccessible(true);
            return (Stage<?>) method.invoke(null, tEnv, path);
        } catch (NoSuchMethodException e) {
            String methodName =
                    String.format("%s::load(StreamTableEnvironment, String)", className);
            throw new RuntimeException(
                    "Failed to load stage because the static method "
                            + methodName
                            + " is not implemented.",
                    e);
        } catch (ClassNotFoundException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException("Failed to load stage.", e);
        }
    }

    /**
     * Saves the model data stream to the given path using the model encoder.
     *
     * @param model The model data stream.
     * @param path The parent directory of the model data file.
     * @param modelEncoder The encoder to encode the model data.
     * @param <T> The class type of the model data.
     */
    public static <T> void saveModelData(
            DataStream<T> model, String path, Encoder<T> modelEncoder) {
        FileSink<T> sink =
                FileSink.forRowFormat(
                                new org.apache.flink.core.fs.Path(getDataPath(path)), modelEncoder)
                        .withRollingPolicy(OnCheckpointRollingPolicy.build())
                        .withBucketAssigner(new BasePathBucketAssigner<>())
                        .build();
        model.sinkTo(sink);
    }

    /**
     * Loads the model data from the given path using the model decoder.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path The parent directory of the model data file.
     * @param modelDecoder The decoder used to decode the model data.
     * @param <T> The class type of the model data.
     * @return The loaded model data.
     */
    public static <T> Table loadModelData(
            StreamTableEnvironment tEnv, String path, SimpleStreamFormat<T> modelDecoder) {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);
        Source<T, ?, ?> source =
                FileSource.forRecordStreamFormat(modelDecoder, new Path(getDataPath(path))).build();
        DataStream<T> modelDataStream =
                env.fromSource(source, WatermarkStrategy.noWatermarks(), "modelData");
        return tEnv.fromDataStream(modelDataStream);
    }
}
