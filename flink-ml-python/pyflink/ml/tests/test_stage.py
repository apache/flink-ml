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
import os
from typing import Dict, Any

from pyflink.common import Time
from pyflink.table import StreamTableEnvironment

from pyflink.ml.api import Stage
from pyflink.ml.linalg import Vectors
from pyflink.ml.param import ParamValidators, Param, BooleanParam, IntParam, \
    FloatParam, StringParam, VectorParam, IntArrayParam, FloatArrayParam, StringArrayParam, \
    WindowsParam
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase

from pyflink.ml.common.window import ProcessingTimeSessionWindows, GlobalWindows, \
    CountTumblingWindows, EventTimeTumblingWindows, ProcessingTimeTumblingWindows, \
    EventTimeSessionWindows

BOOLEAN_PARAM = BooleanParam("boolean_param", "Description", False)
INT_PARAM = IntParam("int_param", "Description", 1, ParamValidators.lt(100))
FLOAT_PARAM = FloatParam("float_param", "Description", 3.0, ParamValidators.lt(100))
STRING_PARAM = StringParam('string_param', "Description", "5")
INT_ARRAY_PARAM = IntArrayParam("int_array_param", "Description", (6, 7))
FLOAT_ARRAY_PARAM = FloatArrayParam("float_array_param", "Description", (10.0, 11.0))
STRING_ARRAY_PARAM = StringArrayParam("string_array_param", "Description", ("14", "15"))
VECTOR_PARAM = VectorParam('vector_param', "Description", Vectors.dense(1, 2, 3))
WINDOWS_PARAM = WindowsParam('windows_param', "Description",
                             ProcessingTimeSessionWindows.with_gap(Time.milliseconds(100)))
EXTRA_INT_PARAM = IntParam("extra_int_param",
                           "Description",
                           20,
                           ParamValidators.always_true())
PARAM_WITH_NONE_DEFAULT = IntParam("param_with_none_default",
                                   "Must be explicitly set with a non-none value",
                                   None,
                                   ParamValidators.not_null())


class StageTest(PyFlinkMLTestCase):

    def test_param_set_value_with_name(self):
        stage = MyStage()
        stage.set(INT_PARAM, 2)
        self.assertEqual(2, stage.get(INT_PARAM))

        dense_vec = Vectors.dense(2, 2)
        stage.set(VECTOR_PARAM, dense_vec)
        self.assertEqual(dense_vec.get(0), stage.get(VECTOR_PARAM).get(0))
        self.assertEqual(dense_vec.get(1), stage.get(VECTOR_PARAM).get(1))

        sparse_vec = Vectors.sparse(3, [0, 2], [2, 2])
        stage.set(VECTOR_PARAM, sparse_vec)
        self.assertEqual(sparse_vec.get(0), stage.get(VECTOR_PARAM).get(0))
        self.assertEqual(sparse_vec.get(1), stage.get(VECTOR_PARAM).get(1))
        self.assertEqual(sparse_vec.get(2), stage.get(VECTOR_PARAM).get(2))

        param = stage.get_param("int_param")
        stage.set(param, 3)
        self.assertEqual(3, stage.get(param))

        param = stage.get_param('extra_int_param')
        stage.set(param, 50)
        self.assertEqual(50, stage.get(param))

    def test_param_with_null_default(self):
        stage = MyStage()
        import pytest
        with pytest.raises(ValueError, match='value should not be None'):
            stage.get(PARAM_WITH_NONE_DEFAULT)

        stage.set(PARAM_WITH_NONE_DEFAULT, 3)
        self.assertEqual(3, stage.get(PARAM_WITH_NONE_DEFAULT))

    def test_param_set_invalid_value(self):
        stage = MyStage()
        import pytest

        with pytest.raises(ValueError, match='Parameter int_param is given an invalid value 100.'):
            stage.set(INT_PARAM, 100)

        with pytest.raises(ValueError,
                           match='Parameter float_param is given an invalid value 100.0.'):
            stage.set(FLOAT_PARAM, 100.0)

        with pytest.raises(TypeError,
                           match="Parameter int_param's type <class 'int'> is incompatible with "
                                 "the type of <class 'str'>"):
            stage.set(INT_PARAM, "100")

        with pytest.raises(TypeError,
                           match="Parameter string_param's type <class 'str'> is incompatible with"
                                 " the type of <class 'int'>"):
            stage.set(STRING_PARAM, 100)

        with pytest.raises(TypeError,
                           match="Parameter vector_param's type <class 'pyflink.ml.linalg"
                                 ".Vector'> is incompatible with the type of <class 'int'>"):
            stage.set(VECTOR_PARAM, 100)

        with pytest.raises(TypeError,
                           match="Parameter windows_param's type <class 'pyflink.ml.common.window"
                                 ".Windows'> is incompatible with the type of <class 'int'>"):
            stage.set(WINDOWS_PARAM, 100)

    def test_param_set_valid_value(self):
        stage = MyStage()

        stage.set(BOOLEAN_PARAM, True)
        self.assertTrue(stage.get(BOOLEAN_PARAM))

        stage.set(INT_PARAM, 50)
        self.assertEqual(50, stage.get(INT_PARAM))

        stage.set(FLOAT_PARAM, 50.0)
        self.assertEqual(50.0, stage.get(FLOAT_PARAM))

        stage.set(STRING_PARAM, "50")
        self.assertEqual("50", stage.get(STRING_PARAM))

        dense_vec = Vectors.dense(2, 2)
        stage.set(VECTOR_PARAM, dense_vec)
        self.assertEqual(dense_vec.get(0), stage.get(VECTOR_PARAM).get(0))
        self.assertEqual(dense_vec.get(1), stage.get(VECTOR_PARAM).get(1))

        sparse_vec = Vectors.sparse(3, [0, 2], [2, 2])
        stage.set(VECTOR_PARAM, sparse_vec)
        self.assertEqual(sparse_vec.get(0), stage.get(VECTOR_PARAM).get(0))
        self.assertEqual(sparse_vec.get(1), stage.get(VECTOR_PARAM).get(1))
        self.assertEqual(sparse_vec.get(2), stage.get(VECTOR_PARAM).get(2))

        windows_list = [
            GlobalWindows(),
            CountTumblingWindows.of(100),
            EventTimeTumblingWindows.of(Time.milliseconds(100)),
            ProcessingTimeTumblingWindows.of(Time.seconds(100)),
            EventTimeSessionWindows.with_gap(Time.minutes(100)),
            ProcessingTimeSessionWindows.with_gap(Time.hours(100))
        ]

        for windows in windows_list:
            stage.set(WINDOWS_PARAM, windows)
            self.assertEqual(windows, stage.get(WINDOWS_PARAM))

        stage.set(INT_ARRAY_PARAM, (50, 51))
        self.assertEqual((50, 51), stage.get(INT_ARRAY_PARAM))

        stage.set(FLOAT_ARRAY_PARAM, (50.0, 51.0))
        self.assertEqual((50.0, 51.0), stage.get(FLOAT_ARRAY_PARAM))

        stage.set(STRING_ARRAY_PARAM, ("50", "51"))
        self.assertEqual(("50", "51"), stage.get(STRING_ARRAY_PARAM))

    def test_save_load_windows_params(self):
        stage = MyStage()
        stage.set(PARAM_WITH_NONE_DEFAULT, 1)

        windows_list = [
            GlobalWindows(),
            CountTumblingWindows.of(100),
            EventTimeTumblingWindows.of(Time.milliseconds(100)),
            ProcessingTimeTumblingWindows.of(Time.seconds(100)),
            EventTimeSessionWindows.with_gap(Time.minutes(100)),
            ProcessingTimeSessionWindows.with_gap(Time.hours(100))
        ]

        for windows in windows_list:
            stage.set(WINDOWS_PARAM, windows)
            path = os.path.join(self.temp_dir, "test_save_load_windows_params" + str(windows))
            stage.save(path)
            loaded_stage = MyStage.load(self.env, path)
            self.assertEqual(windows, loaded_stage.get(WINDOWS_PARAM))

    def test_stage_save_load(self):
        stage = MyStage()
        stage.set(PARAM_WITH_NONE_DEFAULT, 1)
        path = os.path.join(self.temp_dir, "test_stage_save_load")
        stage.save(path)
        loaded_stage = MyStage.load(self.env, path)
        self.assertEqual(stage.get_param_map(), loaded_stage.get_param_map())
        self.assertEqual(1, loaded_stage.get(INT_PARAM))

    def test_validators(self):
        gt = ParamValidators.gt(10)
        self.assertFalse(gt.validate(None))
        self.assertFalse(gt.validate(5))
        self.assertFalse(gt.validate(10))
        self.assertTrue(gt.validate(15))

        gt_eq = ParamValidators.gt_eq(10)
        self.assertFalse(gt_eq.validate(None))
        self.assertFalse(gt_eq.validate(5))
        self.assertTrue(gt_eq.validate(10))
        self.assertTrue(gt_eq.validate(15))

        lt = ParamValidators.lt(10)
        self.assertFalse(lt.validate(None))
        self.assertTrue(lt.validate(5))
        self.assertFalse(lt.validate(10))
        self.assertFalse(lt.validate(15))

        lt_eq = ParamValidators.lt_eq(10)
        self.assertFalse(lt_eq.validate(None))
        self.assertTrue(lt_eq.validate(5))
        self.assertTrue(lt_eq.validate(10))
        self.assertFalse(lt_eq.validate(15))

        in_range_inclusive = ParamValidators.in_range(5, 15)
        self.assertFalse(in_range_inclusive.validate(None))
        self.assertFalse(in_range_inclusive.validate(0))
        self.assertTrue(in_range_inclusive.validate(5))
        self.assertTrue(in_range_inclusive.validate(10))
        self.assertTrue(in_range_inclusive.validate(15))
        self.assertFalse(in_range_inclusive.validate(20))

        in_range_exclusive = ParamValidators.in_range(5, 15, False, False)
        self.assertFalse(in_range_exclusive.validate(None))
        self.assertFalse(in_range_exclusive.validate(0))
        self.assertFalse(in_range_exclusive.validate(5))
        self.assertTrue(in_range_exclusive.validate(10))
        self.assertFalse(in_range_exclusive.validate(15))
        self.assertFalse(in_range_exclusive.validate(20))

        in_array = ParamValidators.in_array([1, 2, 3])
        self.assertFalse(in_array.validate(None))
        self.assertTrue(in_array.validate(1))
        self.assertFalse(in_array.validate(0))

        not_null = ParamValidators.not_null()
        self.assertTrue(not_null.validate(5))
        self.assertFalse(not_null.validate(None))


class MyStage(Stage):
    def __init__(self):
        self._param_map = {}  # type: Dict[Param, Any]
        self._init_param()

    def save(self, path: str) -> None:
        from pyflink.ml.util import read_write_utils
        read_write_utils.save_metadata(self, path)

    @classmethod
    def load(cls, t_env: StreamTableEnvironment, path: str):
        from pyflink.ml.util import read_write_utils
        return read_write_utils.load_stage_param(path)

    def get_param_map(self) -> Dict['Param[Any]', Any]:
        return self._param_map

    def _init_param(self):
        self._param_map[BOOLEAN_PARAM] = BOOLEAN_PARAM.default_value
        self._param_map[INT_PARAM] = INT_PARAM.default_value
        self._param_map[FLOAT_PARAM] = FLOAT_PARAM.default_value
        self._param_map[STRING_PARAM] = STRING_PARAM.default_value
        self._param_map[VECTOR_PARAM] = VECTOR_PARAM.default_value
        self._param_map[WINDOWS_PARAM] = WINDOWS_PARAM.default_value
        self._param_map[INT_ARRAY_PARAM] = INT_ARRAY_PARAM.default_value
        self._param_map[FLOAT_ARRAY_PARAM] = FLOAT_ARRAY_PARAM.default_value
        self._param_map[STRING_ARRAY_PARAM] = STRING_ARRAY_PARAM.default_value
        self._param_map[EXTRA_INT_PARAM] = EXTRA_INT_PARAM.default_value
        self._param_map[PARAM_WITH_NONE_DEFAULT] = PARAM_WITH_NONE_DEFAULT.default_value
