	v��2Sr7@v��2Sr7@!v��2Sr7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-v��2Sr7@m6Vb�Q4@1� ��C�?A�I+��?I 7��@*	�����F@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate������?!E	XD	XC@)2U0*��?19��8��A@:Preprocessing2U
Iterator::Model::ParallelMapV2�I+��?!m
�l
�8@)�I+��?1m
�l
�8@:Preprocessing2F
Iterator::Model�o_��?!�2��2�B@)�+e�Xw?1����)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��0�*x?!�c��c�*@)����Mbp?1�{�{"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)\���(�?!��O@)_�Q�k?1O��N��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!Y�1Y�1@)ŏ1w-!_?1Y�1Y�1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�&S��?!������D@)/n��R?1$��#��@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!�����?)-C��6J?1�����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!V.�U.��?)a2U0*�C?1V.�U.��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�12.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIC}���X@Q=�`y�W�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	m6Vb�Q4@m6Vb�Q4@!m6Vb�Q4@      ��!       "	� ��C�?� ��C�?!� ��C�?*      ��!       2	�I+��?�I+��?!�I+��?:	 7��@ 7��@! 7��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qC}���X@y=�`y�W�?