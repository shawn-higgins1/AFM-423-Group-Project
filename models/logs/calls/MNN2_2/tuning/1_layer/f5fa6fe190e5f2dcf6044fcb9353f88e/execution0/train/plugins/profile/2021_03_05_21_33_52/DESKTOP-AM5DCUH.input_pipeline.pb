		6��{5@	6��{5@!	6��{5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-	6��{5@$0��'3@1����n�?A��0�*�?I�@� @*	������F@2U
Iterator::Model::ParallelMapV2�{�Pk�?!��B:<@)�{�Pk�?1��B:<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat'�����?!XR���i7@)/n���?1�2�\�A3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate������?!xڪ�b9@)vq�-�?1ڪ�bI1@:Preprocessing2F
Iterator::Model�N@aÓ?!�3�t�E@)-C��6z?1����=,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice���_vOn?!;_��K1 @)���_vOn?1;_��K1 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipF%u��?!w�{�O�L@)�����g?1xڪ�b@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!z} �T�@)ŏ1w-!_?1z} �T�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!����=<@)a2U0*�S?1���6�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIXa�{�X@Q(T����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	$0��'3@$0��'3@!$0��'3@      ��!       "	����n�?����n�?!����n�?*      ��!       2	��0�*�?��0�*�?!��0�*�?:	�@� @�@� @!�@� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qXa�{�X@y(T����?