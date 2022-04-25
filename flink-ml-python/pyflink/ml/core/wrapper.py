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
from typing import List, Dict, Any

from pyflink.java_gateway import get_gateway
from pyflink.table import Table, StreamTableEnvironment
from pyflink.util.java_utils import to_jarray

from pyflink.ml.core.api import Model, Transformer, AlgoOperator, Stage, Estimator
from pyflink.ml.core.param import Param, WithParams


class JavaWrapper(ABC):
    """
    Wrapper class for a Java object
    """

    def __init__(self, java_obj):
        self._java_obj = java_obj


class JavaWithParams(WithParams, JavaWrapper):
    """
    Wrapper class for a Java WithParams.
    """
    PYTHON_PARAM_NAME_TO_JAVA_PARM_NAME = {
        'distance_measure': 'distanceMeasure',
        'features_col': 'featuresCol',
        'global_batch_size': 'globalBatchSize',
        'handle_invalid': 'handleInvalid',
        'input_cols': 'inputCols',
        'label_col': 'labelCol',
        'learning_rate': 'learningRate',
        'max_iter': 'maxIter',
        'multi_class': 'multiClass',
        'output_cols': 'outputCols',
        'prediction_col': 'predictionCol',
        'raw_prediction_col': 'rawPredictionCol',
        'reg': 'reg',
        'seed': 'seed',
        'tol': 'tol',
        'weight_col': 'weightCol',
        'k': 'k',
        'model_type': 'modelType',
        'smoothing': 'smoothing',
        'init_mode': 'initMode',
        'batch_strategy': 'batchStrategy',
        'decay_factor': 'decayFactor'
    }

    def __init__(self, java_params):
        super(JavaWithParams, self).__init__(java_params)

    def set(self, param: Param, value) -> WithParams:
        java_param_name = self._to_java_param_name(param.name)
        set_method_name = ''.join(['set', java_param_name[0].upper(), java_param_name[1:]])
        getattr(self._java_obj, set_method_name)(value)
        return self

    def get(self, param: Param):
        java_param_name = self._to_java_param_name(param.name)
        get_method_name = ''.join(['get', java_param_name[0].upper(), java_param_name[1:]])
        return getattr(self._java_obj, get_method_name)()

    def get_param_map(self) -> Dict[Param, Any]:
        return self._java_obj.getParamMap()

    def _to_java_param_name(self, name):
        if name in self.PYTHON_PARAM_NAME_TO_JAVA_PARM_NAME:
            return self.PYTHON_PARAM_NAME_TO_JAVA_PARM_NAME[name]
        else:
            raise Exception('Unknown param exception %s' % name)


class JavaStage(Stage, JavaWithParams, ABC):
    """
    Wrapper class for a Java Stage.
    """

    def __init__(self, java_stage):
        super(JavaStage, self).__init__(java_stage)

    def save(self, path: str) -> None:
        self._java_obj.save(path)


class JavaAlgoOperator(AlgoOperator, JavaStage, ABC):
    """
    Wrapper class for a Java AlgoOperator.
    """

    def __init__(self, java_algo_operator):
        super(JavaAlgoOperator, self).__init__(java_algo_operator)

    def transform(self, *inputs: Table) -> List[Table]:
        results = self._java_obj.transform(_to_java_tables(*inputs))
        return [Table(t, inputs[0]._t_env) for t in results]


class JavaTransformer(Transformer, JavaAlgoOperator, ABC):
    """
    Wrapper class for a Java Transformer.
    """

    def __init__(self, java_transformer):
        super(JavaTransformer, self).__init__(java_transformer)


class JavaModel(Model, JavaTransformer, ABC):
    """
    Wrapper class for a Java Model.
    """

    def __init__(self, java_model):
        if java_model is None:
            super(JavaModel, self).__init__(_to_java_reference(self._java_model_path())())
        else:
            super(JavaModel, self).__init__(java_model)
        self._t_env = None

    def set_model_data(self, *inputs: Table) -> Model:
        self._t_env = inputs[0]._t_env
        self._java_obj.setModelData(_to_java_tables(*inputs))
        return self

    def get_model_data(self) -> List[Table]:
        return [Table(t, self._t_env) for t in self._java_obj.getModelData()]

    @classmethod
    def load(cls, t_env: StreamTableEnvironment, path: str):
        java_model = _to_java_reference(cls._java_model_path()).load(t_env._j_tenv, path)
        instance = cls(java_model)
        return instance

    @classmethod
    @abstractmethod
    def _java_model_path(cls) -> str:
        pass


class JavaEstimator(Estimator, JavaStage, ABC):
    """
    Wrapper class for a Java Estimator.
    """

    def __init__(self):
        super(JavaEstimator, self).__init__(_new_java_obj(self._java_estimator_path()))

    def fit(self, *inputs: Table) -> Model:
        return self._create_model(self._java_obj.fit(_to_java_tables(*inputs)))

    @classmethod
    def _create_model(cls, java_model) -> Model:
        """
        Creates a model from the input Java model reference.
        """
        pass

    @classmethod
    def load(cls, t_env: StreamTableEnvironment, path: str):
        """
        Instantiates a new stage instance based on the data read from the given path.
        """
        java_estimator = _to_java_reference(cls._java_estimator_path()).load(t_env._j_tenv, path)
        instance = cls()
        instance._java_obj = java_estimator
        return instance

    @classmethod
    @abstractmethod
    def _java_estimator_path(cls) -> str:
        pass


def _to_java_reference(java_class: str):
    java_obj = get_gateway().jvm
    for name in java_class.split("."):
        java_obj = getattr(java_obj, name)
    return java_obj


def _new_java_obj(java_class: str, *java_args):
    """
    Returns a new Java object.
    """
    java_obj = _to_java_reference(java_class)
    return java_obj(*java_args)


def _to_java_tables(*inputs: Table):
    """
    Converts Python Tables to Java tables.
    """
    gateway = get_gateway()
    return to_jarray(gateway.jvm.org.apache.flink.table.api.Table, [t._j_table for t in inputs])
