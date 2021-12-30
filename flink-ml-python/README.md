Flink ML is a library which provides machine learning (ML) APIs and infrastructures that simplify the building of ML pipelines. Users can implement ML algorithms with the standard ML APIs and further use these infrastructures to build ML pipelines for both training and inference jobs.

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
