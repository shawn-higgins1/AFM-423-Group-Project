	=,Ԛ�5@=,Ԛ�5@!=,Ԛ�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-=,Ԛ�5@�� �Sn3@1��<���?AM�St$�?I��� 8�?*	fffff&I@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate���&�?!$ϟ꥖B@)�� �rh�?1O�b2�@@:Preprocessing2U
Iterator::Model::ParallelMapV2_�Qڋ?!L-у�	;@)_�Qڋ?1L-у�	;@:Preprocessing2F
Iterator::Model{�G�z�?!V��v�C@)-C��6z?1t��N�r)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq{?!V��v�*@)a2U0*�s?1>*{�#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�sF���?!��^j�N@)F%u�k?1aZi>@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�����7@)ŏ1w-!_?1�����7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapDio��ɔ?!����-D@)-C��6Z?1t��N�r	@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!Ҽ�"$��?)����MbP?1Ҽ�"$��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!Ɲ�2D�?)Ǻ���F?1Ɲ�2D�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIB����~X@Q˷�GA+ @Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�� �Sn3@�� �Sn3@!�� �Sn3@      ��!       "	��<���?��<���?!��<���?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	��� 8�?��� 8�?!��� 8�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qB����~X@y˷�GA+ @