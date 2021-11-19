Flink ML is a library which provides machine learning (ML) APIs and libraries that simplify the building of machine learning pipelines. It provides a set of standard ML APIs for MLlib developers to implement ML algorithms, as well as libraries of ML algorithms that can be used to build ML pipelines for both training and inference jobs.

Flink ML is developed under the umbrella of [Apache Flink](https://flink.apache.org/).

## <a name="build"></a>Python Packaging

Prerequisites for building apache-flink-ml:

* Unix-like environment (we use Linux, Mac OS X)
* Python version(3.6, 3.7 or 3.8) is required

Then go to the root directory of flink-ml-python source code and run this command to build the sdist package of apache-flink-ml:
```bash
cd flink-ml-python; python setup.py sdist;
```

The sdist package of apache-flink-ml will be found under ./flink-ml-python/dist/. It could be used for installation, such as:
```bash
python -m pip install dist/*.tar.gz
```
