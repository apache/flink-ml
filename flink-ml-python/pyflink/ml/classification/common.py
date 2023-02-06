################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
from abc import ABC, abstractmethod

from pyflink.ml.wrapper import JavaModel, JavaEstimator

JAVA_CLASSIFICATION_PACKAGE_NAME = "org.apache.flink.ml.classification"


class JavaClassificationModel(JavaModel, ABC):
    """
    Wrapper class for a Java Classification Model.
    """

    def __init__(self, java_model):
        super(JavaClassificationModel, self).__init__(java_model)

    @classmethod
    def _java_stage_path(cls) -> str:
        return ".".join(
            [JAVA_CLASSIFICATION_PACKAGE_NAME,
             cls._java_model_package_name(),
             cls._java_model_class_name()])

    @classmethod
    @abstractmethod
    def _java_model_package_name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def _java_model_class_name(cls) -> str:
        pass


class JavaClassificationEstimator(JavaEstimator, ABC):
    """
    Wrapper class for a Java Classification Estimator.
    """

    def __init__(self):
        super(JavaClassificationEstimator, self).__init__()

    @classmethod
    def _java_stage_path(cls):
        return ".".join(
            [JAVA_CLASSIFICATION_PACKAGE_NAME,
             cls._java_estimator_package_name(),
             cls._java_estimator_class_name()])

    @classmethod
    @abstractmethod
    def _java_estimator_package_name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def _java_estimator_class_name(cls) -> str:
        pass
