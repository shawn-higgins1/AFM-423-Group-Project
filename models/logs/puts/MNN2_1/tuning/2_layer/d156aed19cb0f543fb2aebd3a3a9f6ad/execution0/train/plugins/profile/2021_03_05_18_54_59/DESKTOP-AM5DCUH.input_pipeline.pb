	t~����7@t~����7@!t~����7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-t~����7@6��g�4@1遏��S�?A9��m4��?Iur��@*	fffff�J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenatej�t��?!S�:+D@)��ׁsF�?1�����B@:Preprocessing2U
Iterator::Model::ParallelMapV2�
F%u�?!��P���7@)�
F%u�?1��P���7@:Preprocessing2F
Iterator::Model�ݓ��Z�?!��6ֺA@)a��+ey?1�b�C'@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat� �	�?!�m�u��,@)�����w?1��o{�%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip"��u���?!�u��"P@)ŏ1w-!o?1����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!����@)ŏ1w-!_?1����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapHP�s�?!A��o{E@)Ǻ���V?1��\@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!�쇑��?)����MbP?1�쇑��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!��\�?)Ǻ���F?1��\�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIw$�a��X@QC����^�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	6��g�4@6��g�4@!6��g�4@      ��!       "	遏��S�?遏��S�?!遏��S�?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	ur��@ur��@!ur��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qw$�a��X@yC����^�?