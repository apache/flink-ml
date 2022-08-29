Flink ML is a library which provides machine learning (ML) APIs and
infrastructures that simplify the building of ML pipelines. Users can implement
ML algorithms with the standard ML APIs and further use these infrastructures to
build ML pipelines for both training and inference jobs.

Flink ML is developed under the umbrella of [Apache
Flink](https://flink.apache.org/).

## <a name="start"></a>Getting Started

You can follow this [quick
start](https://nightlies.apache.org/flink/flink-ml-docs-master/docs/try-flink-ml/java/quick-start/)
guideline to get hands-on experience with Flink ML.

## <a name="build"></a>Building the Project

Run the `mvn clean package` command.

Then you will find a JAR file that contains your application, plus any libraries
that you may have added as dependencies to the application:
`target/<artifact-id>-<version>.jar`.

## <a name="benchmark"></a>Benchmark

Flink ML provides functionalities to benchmark its machine learning algorithms.
For detailed information, please check the [Benchmark Getting
Started](./flink-ml-benchmark/README.md).

## <a name="documentation"></a>Documentation

The documentation of Flink ML is located on the website:
https://nightlies.apache.org/flink/flink-ml-docs-master/ or in the docs/
directory of the source code.

## <a name="contributing"></a>Contributing

You can learn more about how to contribute in the [Apache Flink
website](https://flink.apache.org/contributing/how-to-contribute.html). For code
contributions, please read carefully the [Contributing
Code](https://flink.apache.org/contributing/contribute-code.html) section for an
overview of ongoing community work.

## <a name="license"></a>License

The code in this repository is licensed under the [Apache Software License
2](LICENSE).
