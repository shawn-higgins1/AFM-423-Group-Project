	�bE��1@�bE��1@!�bE��1@	����3�?����3�?!����3�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�bE��1@e��~��0@1y�|?�?AxԘsI�?I[z4�S�?Y���b('z?*	fffff&I@2U
Iterator::Model::ParallelMapV2 �o_Ή?!~q�A�9@) �o_Ή?1~q�A�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-��?!#]���<@)��@��ǈ?1��� @8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�Pk�w�?!<��Z�;@)/n���?1@<F�~1@:Preprocessing2F
Iterator::Model�ݓ��Z�?!�=��B@) �o_�y?1~q�A�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��ZӼ�t?!��ܢ@G$@)��ZӼ�t?1��ܢ@G$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2U0*��?!�j�u6O@)�~j�t�h?1���[�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!>*{�@)a2U0*�c?1>*{�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����3�?In���X@Qt 47��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	e��~��0@e��~��0@!e��~��0@      ��!       "	y�|?�?y�|?�?!y�|?�?*      ��!       2	xԘsI�?xԘsI�?!xԘsI�?:	[z4�S�?[z4�S�?![z4�S�?B      ��!       J	���b('z?���b('z?!���b('z?R      ��!       Z	���b('z?���b('z?!���b('z?b      ��!       JGPUY����3�?b qn���X@yt 47��?