	=ڨN�5@=ڨN�5@!=ڨN�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-=ڨN�5@�V�9�y3@1��b�D�?A��ͪ�զ?I��\4d� @*	     @J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��~j�t�?!�a�aB@)r�����?1�m۶m�@@:Preprocessing2U
Iterator::Model::ParallelMapV2%u��?!     <@)%u��?1     <@:Preprocessing2F
Iterator::Model46<�R�?!1�0�D@)�ZӼ�}?1�0�0+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty�&1�|?!������*@){�G�zt?1�0�0#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����o�?!��<��<M@)�����g?1�a�a@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!��y��y@)����Mb`?1��y��y@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_�L�?!�<��<�C@)��H�}]?1ܶm۶m@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorǺ���F?!VUUUUU�?)Ǻ���F?1VUUUUU�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�$I�$I�?)a2U0*�C?1�$I�$I�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�ر/��X@Q��'h0�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�V�9�y3@�V�9�y3@!�V�9�y3@      ��!       "	��b�D�?��b�D�?!��b�D�?*      ��!       2	��ͪ�զ?��ͪ�զ?!��ͪ�զ?:	��\4d� @��\4d� @!��\4d� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�ر/��X@y��'h0�?