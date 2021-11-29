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
import pickle
import time
from importlib import import_module
from typing import List, Dict, Any

import cloudpickle
from pyflink.datastream import StreamExecutionEnvironment

from pyflink.ml.core.api import Stage


def save_pipeline(pipeline: Stage, stages: List[Stage], path: str) -> None:
    """
    Saves a Pipeline or PipelineModel with the given list of stages to the given path.

    :param pipeline: A Pipeline or PipelineModel instance.
    :param stages: A list of stages of the given pipeline.
    :param path: The parent directory to save the pipeline metadata and its stages.
    """
    # Creates parent directories if not already created.
    os.makedirs(path, exist_ok=True)

    extra_metadata = {'num_stages': len(stages)}
    save_metadata(pipeline, path, extra_metadata)

    num_stages = len(stages)
    for i, stage in enumerate(stages):
        stage_path = get_path_for_pipeline_stage(i, num_stages, path)
        stage.save(stage_path)


def load_pipeline(env: StreamExecutionEnvironment, path: str) -> List[Stage]:
    """
    Loads the stages of a Pipeline or PipelineModel from the given path.

    :param env: A StreamExecutionEnvironment instance.
    :param path: A StreamExecutionEnvironment instance.
    :return: A list of stages.
    """
    meta_data = load_metadata(path)
    num_stages = meta_data['num_stages']
    return [load_stage(env, get_path_for_pipeline_stage(i, num_stages, path))
            for i in range(num_stages)]


def save_metadata(stage: Stage, path: str, extra_metadata=None) -> None:
    """
    Saves the metadata of the given stage and the extra metadata to a file named `metadata` under
    the given path. The metadata of a stage includes the stage class name, parameter values etc.

    Required: the metadata file under the given path should not exist.

    :param stage: The stage instance.
    :param path: The parent directory to save the stage metadata.
    :param extra_metadata: The extra metadata to be saved.
    """
    if extra_metadata is None:
        extra_metadata = {}
    os.makedirs(path, exist_ok=True)

    metadata = {k: v for k, v in extra_metadata.items()}
    metadata['module_name'] = str(stage.__module__)
    metadata['class_name'] = str(type(stage).__name__)
    metadata['timestamp'] = time.time()
    metadata['param_map'] = {cloudpickle.dumps(k): k.json_encode(v)
                             for k, v in stage.get_param_map().items()}
    # TODO: add version in the metadata.

    metadata_bytes = pickle.dumps(metadata)
    metadata_path = os.path.join(path, 'metadata')
    if os.path.isfile(metadata_path):
        raise IOError(f'File {metadata_path} already exists.')
    with open(metadata_path, 'wb') as fd:
        fd.write(metadata_bytes)


def load_metadata(path: str) -> Dict[str, Any]:
    """
    Loads the metadata from the metadata file under the given path.

    :param path: The parent directory of the metadata file to read from.
    :return: A Dict from metadata name to metadata value.
    """
    metadata_path = os.path.join(path, "metadata")
    with open(metadata_path, 'rb') as fd:
        metadata_bytes = fd.read()
    meta_data = pickle.loads(metadata_bytes)
    return meta_data


def load_stage(env: StreamExecutionEnvironment, path: str) -> Stage:
    """
    Loads the stage from the given path by invoking the static load() method of the stage. The
    stage module name and class name are read from the metadata file under the given path. The
    load() method is expected to construct the stage instance with the saved parameters, model data
    and other metadata if exists.

    :param env: A StreamExecutionEnvironment instance.
    :param path: The parent directory of the stage metadata file.
    :return: An instance of Stage.
    """
    metadata = load_metadata(path)
    module_name = metadata.get('module_name')
    class_name = metadata.get('class_name')
    stage_class = getattr(import_module(module_name), class_name)
    return stage_class.load(env, path)


def load_stage_param(path: str) -> Stage:
    """
    Loads the stage with the saved parameters from the given path. This method reads the metadata
    file under the given path, instantiates the stage using its no-argument constructor, and
    loads the stage with the param_map from the metadata file.

    Note: This method does not attempt to read model data from the given path. Caller needs to
    read model data from the given path if the stage has model data.

    :param path: The parent directory of the stage metadata file.
    :return: An stage instance.
    """
    metadata = load_metadata(path)
    module_name = metadata.get('module_name')
    class_name = metadata.get('class_name')
    stage_class = getattr(import_module(module_name), class_name)
    stage = stage_class()

    param_map = metadata.get('param_map')
    for k, v in param_map.items():
        param = cloudpickle.loads(k)
        value = param.json_decode(v)
        stage.set(param, value)
    return stage


def get_path_for_pipeline_stage(stage_idx: int, num_stages: int, parent_path: str) -> str:
    """
    Returns a string with value {parent_path}/stages/{stage_idx}, where the stage_idx is prefixed
    with zero or more `0` to have the same length as num_stages. The resulting string can be used
    as the directory to save a stage of the Pipeline or PipelineModel.
    """
    format_str = ("{:0>%sd}" % (num_stages,))
    return os.path.abspath(os.path.join(parent_path, "stages", format_str.format(stage_idx)))
