	�e6Ȱ5@�e6Ȱ5@!�e6Ȱ5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�e6Ȱ5@M�����2@1+2: 	��?Ag��j+��?I���C�@*	     L@2U
Iterator::Model::ParallelMapV2���H�?!I�$I�d<@)���H�?1I�$I�d<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���H�?!I�$I�d<@)��@��ǈ?1�m۶m�5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!I�$I�$6@)��ZӼ�?1m۶m�62@:Preprocessing2F
Iterator::Model��&��?!     �D@)��H�}}?1m۶m۶)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!I�$I�$@)ŏ1w-!o?1I�$I�$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip*��Dؠ?!     `M@)_�Q�k?1�$I�$I@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!۶m۶m@)/n��b?1۶m۶m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!�$I�$�>@)a2U0*�S?1I�$I�$@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�12.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIn��=��X@Q�H���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	M�����2@M�����2@!M�����2@      ��!       "	+2: 	��?+2: 	��?!+2: 	��?*      ��!       2	g��j+��?g��j+��?!g��j+��?:	���C�@���C�@!���C�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qn��=��X@y�H���?