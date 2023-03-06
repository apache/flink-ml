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

import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.fs.Path;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Map;

/** Utility methods for file operations. */
public class FileUtils {

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

    public static FileSystem mkdirs(Path path) throws IOException {
        FileSystem fs = path.getFileSystem();
        fs.mkdirs(path);
        return fs;
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
        Map<String, ?> result = JsonUtils.OBJECT_MAPPER.readValue(buffer.toString(), Map.class);

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
    public static String getPathForPipelineStage(int stageIdx, int numStages, String parentPath) {
        String format =
                String.format("stages%s%%0%dd", File.separator, String.valueOf(numStages).length());
        String fileName = String.format(format, stageIdx);
        return new Path(parentPath, fileName).toString();
    }

    /** Returns a subdirectory of the given path for saving/loading model data. */
    public static Path getDataPath(String path) {
        return new Path(path, "data");
    }
}
