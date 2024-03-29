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
Core API
========

Stage
-----

Base class for a node in a :class:`Pipeline` or :class:`Graph`. The interface is only a concept,
and does not have any actual functionality. Its subclasses could be :class:`Estimator`, :class:`Model`,
:class:`Transformer` or :class:`AlgoOperator`. No other classes should inherit this interface directly.

Each stage is with parameters, and requires a public empty constructor for restoration.

.. currentmodule:: pyflink.ml.api

.. autosummary::
    :toctree: __tmp/

    Stage.save
    Stage.load

AlgoOperator
------------

An :class:`AlgoOperator` takes a list of tables as inputs and produces a list of tables as results.
It can be used to encode generic multi-input multi-output computation logic.

.. currentmodule:: pyflink.ml.api

.. autosummary::
    :toctree: __tmp/

    AlgoOperator.transform

Transformer
-----------

A :class:`Transformer` is an :class:`AlgoOperator` with the semantic difference that it encodes
the transformation logic, such that a record in the output typically corresponds to one record in
the input. In contrast, an :class:`AlgoOperator` is a better fit to express aggregation logic where
a record in the output could be computed from an arbitrary number of records in the input.

.. currentmodule:: pyflink.ml.api

.. autosummary::
    :toctree: __tmp/

    Transformer

Model
-----

A :class:`Model` is typically generated by invoking :func:`~Estimator.fit`. A :class:`Model` is a
:class:`Transformer` with the extra APIs to set and get model data.

.. currentmodule:: pyflink.ml.api

.. autosummary::
    :toctree: __tmp/

    Model.set_model_data
    Model.get_model_data

Estimator
---------

Estimators are responsible for training and generating Models.

.. currentmodule:: pyflink.ml.api

.. autosummary::
    :toctree: __tmp/

    Estimator.fit
