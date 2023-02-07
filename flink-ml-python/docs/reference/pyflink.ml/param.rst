.. ################################################################################
     Licensed to the Apache Software Foundation (ASF) under one
     or more contributor license agreements.  See the NOTICE file
     distributed with this work for additional information
     regarding copyright ownership.  The ASF licenses this file
     to you under the Apache License, Version 2.0 (the
     "License"); you may not use this file except in compliance
     with the License.  You may obtain a copy of the License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
    limitations under the License.
   ################################################################################

=====
Param
=====

WithParams
----------

Interface for classes that take parameters. It provides APIs to set and get parameters.

.. currentmodule:: pyflink.ml.param

.. autosummary::
    :toctree: __tmp/

    WithParams.set
    WithParams.get
    WithParams.get_param
    WithParams.get_param_map

ParamValidator
--------------

An interface to validate that a parameter value is valid.

.. currentmodule:: pyflink.ml.param

.. autosummary::
    :toctree: __tmp/

    ParamValidator.validate

ParamValidators
---------------

Factory methods for common validation functions on numerical values.

.. currentmodule:: pyflink.ml.param

.. autosummary::
    :toctree: __tmp/

    ParamValidators.always_true
    ParamValidators.gt
    ParamValidators.gt_eq
    ParamValidators.lt
    ParamValidators.lt_eq
    ParamValidators.in_range
    ParamValidators.in_array
    ParamValidators.not_null
    ParamValidators.non_empty_array
    ParamValidators.is_sub_set

Params
------

.. currentmodule:: pyflink.ml.param

.. autosummary::
    :toctree: __tmp/

    BooleanParam
    IntParam
    FloatParam
    StringParam
    IntArrayParam
    FloatArrayParam
    FloatArrayArrayParam
    StringArrayParam
    VectorParam
    WindowsParam
