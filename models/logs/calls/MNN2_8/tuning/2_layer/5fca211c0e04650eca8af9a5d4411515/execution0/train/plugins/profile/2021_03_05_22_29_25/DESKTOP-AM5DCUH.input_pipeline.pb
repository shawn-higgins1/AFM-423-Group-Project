	k~��E!7@k~��E!7@!k~��E!7@	C�H�V�?C�H�V�?!C�H�V�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6k~��E!7@�/עE4@1�J�(�?AZd;�O��?I�<�@YH�`���?*	    �H@2U
Iterator::Model::ParallelMapV2y�&1��?!�!1ogH<@)y�&1��?1�!1ogH<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq�?!hH��;@)Ǻ����?1K�Z�R�6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA��ǘ��?!m��&�l6@)����Mb�?1�@�_)0@:Preprocessing2F
Iterator::Model��A�f�?!�q�qE@)lxz�,C|?1�>��+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea��+ei?!�3$��@)a��+ei?1�3$��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���QI�?!8��8��L@)�~j�t�h?1>���>@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!r�q�@)/n��b?1r�q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!���~�8@)/n��R?1r�q�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9C�H�V�?I#� N�X@Q�&�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�/עE4@�/עE4@!�/עE4@      ��!       "	�J�(�?�J�(�?!�J�(�?*      ��!       2	Zd;�O��?Zd;�O��?!Zd;�O��?:	�<�@�<�@!�<�@B      ��!       J	H�`���?H�`���?!H�`���?R      ��!       Z	H�`���?H�`���?!H�`���?b      ��!       JGPUYC�H�V�?b q#� N�X@y�&�����?