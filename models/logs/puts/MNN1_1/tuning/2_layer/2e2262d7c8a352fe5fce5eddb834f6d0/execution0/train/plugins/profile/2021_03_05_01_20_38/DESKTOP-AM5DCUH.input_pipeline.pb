	���|7@���|7@!���|7@	U���,C�?U���,C�?!U���,C�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���|7@t]����4@1�k���P�?A������?I0H���@Y�:8؛�?*	43333�I@2U
Iterator::Model::ParallelMapV2K�=�U�?!�� �z=@)K�=�U�?1�� �z=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�x�	�8@)��_�L�?1b�x�	4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate-C��6�?!�����8@)"��u���?1��� �0@:Preprocessing2F
Iterator::ModelA��ǘ��?!�-��$cE@)lxz�,C|?1����*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!��DZ/ @)�J�4q?1��DZ/ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���B�i�?! �zۜL@)a��+ei?1�N��`�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!��DZ/@)�J�4a?1��DZ/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�!��u��?!����*;@)��_�LU?1b�x�	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9U���,C�?IX��Z�X@QH�I���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	t]����4@t]����4@!t]����4@      ��!       "	�k���P�?�k���P�?!�k���P�?*      ��!       2	������?������?!������?:	0H���@0H���@!0H���@B      ��!       J	�:8؛�?�:8؛�?!�:8؛�?R      ��!       Z	�:8؛�?�:8؛�?!�:8؛�?b      ��!       JGPUYU���,C�?b qX��Z�X@yH�I���?