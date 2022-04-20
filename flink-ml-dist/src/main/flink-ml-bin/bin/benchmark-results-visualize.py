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
import argparse
import json
import matplotlib.pyplot as plt
import re


def get_nested_field_value(nested_fields, field_names):
    for field_name in field_names:
        nested_fields = nested_fields[field_name]
    return nested_fields


def visualize_benchmark_results(file_name, name_pattern, x_field, y_field):
    x_values = []
    y_values = []
    with open(file_name, "r") as config_file:
        for name, config in json.loads(config_file.read()).items():
            if not name_pattern.match(name):
                continue
            x_values.append(get_nested_field_value(config, x_field.split(".")))
            y_values.append(get_nested_field_value(config, y_field.split(".")))

    plt.scatter(x_values, y_values)
    plt.xlabel(x_field)
    plt.ylabel(y_field)
    plt.title("Flink ML Benchmark Results Visualization")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualizes benchmark results.")
    parser.add_argument(
        "file_name", help="Json file to acquire benchmark results.")
    parser.add_argument(
        "--pattern", help="Regex pattern of benchmark names. "
                          "Benchmarks whose names match this pattern will be selected for visualization. "
                          "If not set, all benchmarks will be selected.",
        default=".*")

    field_help_grammar = "Name of the {} field. Use dot(.) to identify nested fields. Default value: `{}`."
    parser.add_argument(
        "--x-field", help=field_help_grammar.format("independent", "inputData.paramMap.numValues"),
        default="inputData.paramMap.numValues")
    parser.add_argument(
        "--y-field", help=field_help_grammar.format("dependent", "results.inputThroughput"),
        default="results.inputThroughput")
    args = parser.parse_args()

    visualize_benchmark_results(args.file_name, re.compile(
        args.pattern), args.x_field, args.y_field)
