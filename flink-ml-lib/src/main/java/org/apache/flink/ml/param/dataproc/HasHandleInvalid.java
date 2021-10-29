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

package org.apache.flink.ml.param.dataproc;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidator;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.WithParams;

/**
 * An interface for classes with a parameter specifying how to deal with invalid tokens.
 */
public interface HasHandleInvalid<T> extends WithParams<T> {
    Param<HandleInvalid> HANDLE_INVALID =
            new HandleInvalidParam(
                    "handleInvalid",
                    "Strategy to handle unseen token when doing prediction, one of \"keep\", \"skip\" or \"error\"",
                    HandleInvalid.KEEP);

    default HandleInvalid getHandleInvalid() {
        return get(HANDLE_INVALID);
    }

    default T setHandleInvalid(HandleInvalid value) {
        return set(HANDLE_INVALID, value);
    }

    default T setHandleInvalid(String value) {
        return set(HANDLE_INVALID, HandleInvalid.valueOf(value));
    }


    /** Class for the handle invalid parameter. */
    class HandleInvalidParam extends Param<HandleInvalid> {

        public HandleInvalidParam(
                String name,
                String description,
                HandleInvalid defaultValue,
                ParamValidator<HandleInvalid> validator) {
            super(name, HandleInvalid.class, description, defaultValue, validator);
        }

        public HandleInvalidParam(String name, String description, HandleInvalid defaultValue) {
            this(name, description, defaultValue, ParamValidators.alwaysTrue());
        }
    }

    /**
     * Strategy to handle unseen token when doing prediction.
     */
    enum HandleInvalid {
        /**
         * Assign "max index" + 1.
         */
        KEEP,
        /**
         * Raise exception.
         */
        ERROR,
        /**
         * Pad with null.
         */
        SKIP
    }
}
