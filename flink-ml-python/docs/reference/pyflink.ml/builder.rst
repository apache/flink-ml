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

========
Pipeline
========

Pipeline
--------

A :class:`Pipeline` acts as an :class:`Estimator`. It consists of an ordered list of stages,
each of which could be an :class:`Estimator`, :class:`Model`, :class:`Transformer` or :class:`AlgoOperator`.

.. currentmodule:: pyflink.ml.builder

.. autosummary::
    :toctree: __tmp/

    Pipeline
    Pipeline.fit
    Pipeline.save
    Pipeline.load
    Pipeline.get_param_map

PipelineModel
-------------

A :class:`PipelineModel` acts as a :class:`Model`. It consists of an ordered list of stages,
each of which could be a :class:`Model`, :class:`Transformer` or :class:`AlgoOperator`.

.. currentmodule:: pyflink.ml.builder

.. autosummary::
    :toctree: __tmp/

    PipelineModel
    PipelineModel.transform
    PipelineModel.save
    PipelineModel.load
    PipelineModel.get_param_map
