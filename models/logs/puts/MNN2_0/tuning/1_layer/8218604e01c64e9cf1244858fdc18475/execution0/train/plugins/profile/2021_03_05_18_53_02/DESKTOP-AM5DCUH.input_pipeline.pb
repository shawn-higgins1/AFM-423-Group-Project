	H��
~6@H��
~6@!H��
~6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-H��
~6@O�)��	4@1�MG 7��?A=�U����?I~��!F@*	     @L@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��JY�8�?!�A�/4C@)���<,�?1$��CoA@:Preprocessing2U
Iterator::Model::ParallelMapV2e�X��?!Y驅��>@)e�X��?1Y驅��>@:Preprocessing2F
Iterator::Model��+e�?!x�!�E@)y�&1�|?1�u��\�(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+ey?!�&_6h�%@);�O��nr?1	�����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?�ܵ�|�?!���~L@)�~j�t�h?1��S+=@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!�!��@)_�Q�[?1�!��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZd;�O��?!ꩅ��ZD@)��_�LU?1f�&_6h@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��_�LU?!f�&_6h@)��_�LU?1f�&_6h@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!+=����?)Ǻ���F?1+=����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�&���X@QND�N���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	O�)��	4@O�)��	4@!O�)��	4@      ��!       "	�MG 7��?�MG 7��?!�MG 7��?*      ��!       2	=�U����?=�U����?!=�U����?:	~��!F@~��!F@!~��!F@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�&���X@yND�N���?