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
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from py4j.java_gateway import JavaObject, get_java_class
from pyflink.common import typeinfo, Time, Row, RowKind
from pyflink.common.typeinfo import _from_java_type, TypeInformation, _is_instance_of, Types, \
    ExternalTypeInfo, RowTypeInfo, TupleTypeInfo
from pyflink.datastream import utils
from pyflink.datastream.utils import pickled_bytes_to_python_converter
from pyflink.java_gateway import get_gateway
from pyflink.table import Table, StreamTableEnvironment, Expression
from pyflink.util.java_utils import to_jarray

from pyflink.ml.api import Model, Transformer, AlgoOperator, Stage, Estimator
from pyflink.ml.linalg import DenseVectorTypeInfo, SparseVectorTypeInfo, DenseMatrixTypeInfo, \
    VectorTypeInfo, DenseVector
from pyflink.ml.param import Param, WithParams, StringArrayParam, IntArrayParam, VectorParam, \
    FloatArrayParam, FloatArrayArrayParam, WindowsParam
from pyflink.ml.common.window import GlobalWindows, CountTumblingWindows, \
    EventTimeTumblingWindows, ProcessingTimeTumblingWindows, EventTimeSessionWindows, \
    ProcessingTimeSessionWindows

_from_java_type_alias = _from_java_type


def _from_java_type_wrapper(j_type_info: JavaObject) -> TypeInformation:
    gateway = get_gateway()
    JGenericTypeInfo = gateway.jvm.org.apache.flink.api.java.typeutils.GenericTypeInfo
    if _is_instance_of(j_type_info, JGenericTypeInfo):
        JClass = j_type_info.getTypeClass()
        if JClass == get_java_class(gateway.jvm.org.apache.flink.ml.linalg.DenseVector):
            return DenseVectorTypeInfo()
        elif JClass == get_java_class(gateway.jvm.org.apache.flink.ml.linalg.SparseVector):
            return SparseVectorTypeInfo()
        elif JClass == get_java_class(gateway.jvm.org.apache.flink.ml.linalg.DenseMatrix):
            return DenseMatrixTypeInfo()
        elif JClass == get_java_class(gateway.jvm.org.apache.flink.ml.linalg.Vector):
            return VectorTypeInfo()
    return _from_java_type_alias(j_type_info)


typeinfo._from_java_type = _from_java_type_wrapper


# TODO: Remove this class after Flink ML depends on a Flink version
#  with FLINK-30168 and FLINK-29477 fixed.
def convert_to_python_obj_wrapper(data, type_info):
    if type_info == Types.PICKLED_BYTE_ARRAY():
        return pickle.loads(data)
    elif isinstance(type_info, ExternalTypeInfo):
        return convert_to_python_obj_wrapper(data, type_info._type_info)
    else:
        gateway = get_gateway()
        pickle_bytes = gateway.jvm.org.apache.flink.ml.python.PythonBridgeUtils. \
            getPickledBytesFromJavaObject(data, type_info.get_java_type_info())
        if isinstance(type_info, RowTypeInfo) or isinstance(type_info, TupleTypeInfo):
            field_data = zip(list(pickle_bytes[1:]), type_info.get_field_types())
            fields = []
            for data, field_type in field_data:
                if len(data) == 0:
                    fields.append(None)
                else:
                    fields.append(pickled_bytes_to_python_converter(data, field_type))
            if isinstance(type_info, RowTypeInfo):
                return Row.of_kind(RowKind(int.from_bytes(pickle_bytes[0], 'little')), *fields)
            else:
                return tuple(fields)
        else:
            return pickled_bytes_to_python_converter(pickle_bytes, type_info)


utils.convert_to_python_obj = convert_to_python_obj_wrapper


class JavaWrapper(ABC):
    """
    Wrapper class for a Java object.
    """

    def __init__(self, java_obj):
        self._java_obj = java_obj


class JavaWithParams(WithParams, JavaWrapper):
    """
    Wrapper class for a Java WithParams.
    """

    def __init__(self, java_params):
        super(JavaWithParams, self).__init__(java_params)

    def set(self, param: Param, value) -> WithParams:
        if type(param) in _map_java_param_converter:
            converter = _map_java_param_converter[type(param)]
        else:
            converter = default_converter
        java_param_name = snake_to_camel(param.name)
        set_method_name = ''.join(['set', java_param_name[0].upper(), java_param_name[1:]])
        getattr(self._java_obj, set_method_name)(converter.to_java(value))
        return self

    def get(self, param: Param):
        if type(param) in _map_java_param_converter:
            converter = _map_java_param_converter[type(param)]
        else:
            converter = default_converter
        java_param_name = snake_to_camel(param.name)
        get_method_name = ''.join(['get', java_param_name[0].upper(), java_param_name[1:]])
        return converter.to_python(getattr(self._java_obj, get_method_name)())

    def get_param_map(self) -> Dict[Param, Any]:
        return self._java_obj.getParamMap()


class JavaStage(Stage, JavaWithParams, ABC):
    """
    Wrapper class for a Java Stage.
    """

    def __init__(self, java_stage):
        super(JavaStage, self).__init__(java_stage)

    def save(self, path: str) -> None:
        self._java_obj.save(path)

    @classmethod
    def load(cls, t_env: StreamTableEnvironment, path: str):
        java_model = _to_java_reference(cls._java_stage_path()).load(t_env._j_tenv, path)
        instance = cls(java_model)
        return instance

    @classmethod
    @abstractmethod
    def _java_stage_path(cls) -> str:
        pass


class JavaAlgoOperator(AlgoOperator, JavaStage, ABC):
    """
    Wrapper class for a Java AlgoOperator.
    """

    def __init__(self, java_algo_operator):
        if java_algo_operator is None:
            super(JavaAlgoOperator, self).__init__(_to_java_reference(self._java_stage_path())())
        else:
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
        super(JavaModel, self).__init__(java_model)
        self._t_env = None

    def set_model_data(self, *inputs: Table) -> Model:
        self._t_env = inputs[0]._t_env
        self._java_obj.setModelData(_to_java_tables(*inputs))
        return self

    def get_model_data(self) -> List[Table]:
        return [Table(t, self._t_env) for t in self._java_obj.getModelData()]


class JavaEstimator(Estimator, JavaStage, ABC):
    """
    Wrapper class for a Java Estimator.
    """

    def __init__(self):
        super(JavaEstimator, self).__init__(_new_java_obj(self._java_stage_path()))

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
        java_estimator = _to_java_reference(cls._java_stage_path()).load(t_env._j_tenv, path)
        instance = cls()
        instance._java_obj = java_estimator
        return instance


class JavaParamConverter(ABC):
    @abstractmethod
    def to_java(self, value):
        pass

    @abstractmethod
    def to_python(self, value):
        pass


class DefaultJavaParamConverter(JavaParamConverter):
    def to_java(self, value):
        return value

    def to_python(self, value):
        return value


class IntArrayJavaPramConverter(JavaParamConverter):
    def to_java(self, value):
        return to_jarray(get_gateway().jvm.java.lang.Integer, value)

    def to_python(self, value):
        return tuple(value[i] for i in range(len(value)))


class FloatArrayJavaPramConverter(JavaParamConverter):
    def to_java(self, value):
        return to_jarray(get_gateway().jvm.java.lang.Double, value)

    def to_python(self, value):
        return tuple(value[i] for i in range(len(value)))


class VectorJavaParamConverter(JavaParamConverter):
    def to_java(self, value):
        jarray = to_jarray(get_gateway().jvm.double, value.to_array())
        return get_gateway().jvm.org.apache.flink.ml.linalg.DenseVector(jarray)

    def to_python(self, value):
        return DenseVector(tuple(value.get(i) for i in range(value.size())))


class WindowsJavaParamConverter(JavaParamConverter):
    @staticmethod
    def _to_java_time(time: Time):
        return get_gateway().jvm.org.apache.flink.api.common.time.Time.milliseconds(
            time.to_milliseconds())

    @staticmethod
    def _to_python_time(time) -> Time:
        return Time.milliseconds(time.toMilliseconds())

    def to_java(self, value):
        java_window_package = get_gateway().jvm.org.apache.flink.ml.common.window
        if isinstance(value, GlobalWindows):
            return java_window_package.GlobalWindows.getInstance()
        elif isinstance(value, CountTumblingWindows):
            return java_window_package.CountTumblingWindows.of(value.size)
        elif isinstance(value, EventTimeTumblingWindows):
            return java_window_package.EventTimeTumblingWindows.of(
                WindowsJavaParamConverter._to_java_time(value.size))
        elif isinstance(value, ProcessingTimeTumblingWindows):
            return java_window_package.ProcessingTimeTumblingWindows.of(
                WindowsJavaParamConverter._to_java_time(value.size))
        elif isinstance(value, EventTimeSessionWindows):
            return java_window_package.EventTimeSessionWindows.withGap(
                WindowsJavaParamConverter._to_java_time(value.gap))
        elif isinstance(value, ProcessingTimeSessionWindows):
            return java_window_package.ProcessingTimeSessionWindows.withGap(
                WindowsJavaParamConverter._to_java_time(value.gap))
        else:
            raise TypeError(f'Python object {str(value)}\' cannot be converted to Java object')

    def to_python(self, value):
        if value.getClass().getName() == \
                "org.apache.flink.ml.common.window.GlobalWindows":
            return GlobalWindows()
        elif value.getClass().getName() == \
                "org.apache.flink.ml.common.window.CountTumblingWindows":
            return CountTumblingWindows.of(value.getSize())
        elif value.getClass().getName() == \
                "org.apache.flink.ml.common.window.EventTimeTumblingWindows":
            return EventTimeTumblingWindows.of(
                WindowsJavaParamConverter._to_python_time(value.getSize()))
        elif value.getClass().getName() == \
                "org.apache.flink.ml.common.window.ProcessingTimeTumblingWindows":
            return ProcessingTimeTumblingWindows.of(
                WindowsJavaParamConverter._to_python_time(value.getSize()))
        elif value.getClass().getName() == \
                "org.apache.flink.ml.common.window.EventTimeSessionWindows":
            return EventTimeSessionWindows.with_gap(
                WindowsJavaParamConverter._to_python_time(value.getGap()))
        elif value.getClass().getName() == \
                "org.apache.flink.ml.common.window.ProcessingTimeSessionWindows":
            return ProcessingTimeSessionWindows.with_gap(
                WindowsJavaParamConverter._to_python_time(value.getGap()))
        else:
            raise TypeError(f'Java object {str(value)}\' cannot be converted to Python object')


class StringArrayJavaParamConverter(JavaParamConverter):
    def to_java(self, value):
        return to_jarray(get_gateway().jvm.java.lang.String, value)

    def to_python(self, value):
        return tuple(value[i] for i in range(len(value)))


class FloatArrayArrayJavaPramConverter(JavaParamConverter):
    def to_java(self, value):
        n = len(value)
        m = len(value[0])
        j_arr = get_gateway().new_array(get_gateway().jvm.java.lang.Double, n, m)
        for i in range(n):
            for j in range(m):
                j_arr[i][j] = value[i][j]
        return j_arr

    def to_python(self, value):
        n = len(value)
        m = len(value[0])
        arr = []
        for i in range(n):
            l = []
            for j in range(m):
                l.append(value[i][j])
            arr.append(tuple(l))
        return tuple(arr)


default_converter = DefaultJavaParamConverter()

_map_java_param_converter = {
    IntArrayParam: IntArrayJavaPramConverter(),
    FloatArrayParam: FloatArrayJavaPramConverter(),
    FloatArrayArrayParam: FloatArrayArrayJavaPramConverter(),
    StringArrayParam: StringArrayJavaParamConverter(),
    VectorParam: VectorJavaParamConverter(),
    WindowsParam: WindowsJavaParamConverter(),
    Param: default_converter
}


def snake_to_camel(method_name):
    output = ''.join(x.capitalize() or '_' for x in method_name.split('_'))
    return output[0].lower() + output[1:]


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


def call_java_table_function(java_table_function_name: str, *args):
    _function = get_gateway().jvm
    for member_name in java_table_function_name.split('.'):
        _function = _function.__getattr__(member_name)
    return Expression(_function(to_jarray(
        get_gateway().jvm.java.lang.Object,
        [expression._j_expr for expression in args])))
