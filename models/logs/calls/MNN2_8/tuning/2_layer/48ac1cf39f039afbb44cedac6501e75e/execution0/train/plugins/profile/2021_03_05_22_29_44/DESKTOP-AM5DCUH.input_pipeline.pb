	��>rk�7@��>rk�7@!��>rk�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��>rk�7@���4@1���߃�?A�H�}�?I��C*@*	�����I@2U
Iterator::Model::ParallelMapV2�ZӼ��?!8g�b�W<@)�ZӼ��?18g�b�W<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�!��u��?!��X�$<@)��@��ǈ?1����&8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�I+��?!O��+��5@)y�&1�|?1ٛ3O��+@:Preprocessing2F
Iterator::Model��A�f�?!�v��D@)S�!�uq{?1�9��*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!��_��@)����Mbp?1��_��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�<,Ԛ�?!��s�G$M@)�����g?1k2fX'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!��_��@)����Mb`?1��_��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!�ur��Y8@)a2U0*�S?1�!�c)@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��f�X@Q}�K���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���4@���4@!���4@      ��!       "	���߃�?���߃�?!���߃�?*      ��!       2	�H�}�?�H�}�?!�H�}�?:	��C*@��C*@!��C*@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��f�X@y}�K���?