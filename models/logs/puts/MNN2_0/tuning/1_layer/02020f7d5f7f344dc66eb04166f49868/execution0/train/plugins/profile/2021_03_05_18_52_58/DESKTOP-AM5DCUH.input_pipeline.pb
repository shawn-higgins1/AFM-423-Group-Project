	�S��!6@�S��!6@!�S��!6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�S��!6@Ǽ�8d�3@1p_�Q�?A,e�X�?I�t{Ic�?*	43333�L@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�~j�t��?!�;����D@)�e��a��?1��L9@C@:Preprocessing2U
Iterator::Model::ParallelMapV2���Q��?!�
�=�!:@)���Q��?1�
�=�!:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��ǘ���?!��L9,@)a��+ey?1��[GP�%@:Preprocessing2F
Iterator::Model��A�f�?!ǿ���4B@)��0�*x?1��ƿ��$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip㥛� ��?!9@Hq_�O@)Ǻ���f?1��18�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�\h{
@)ŏ1w-!_?1�\h{
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!���LF@)-C��6Z?1���L@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensora2U0*�S?!cOy�� @)a2U0*�S?1cOy�� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!��18��?)Ǻ���F?1��18��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI3֮���X@Q��((��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Ǽ�8d�3@Ǽ�8d�3@!Ǽ�8d�3@      ��!       "	p_�Q�?p_�Q�?!p_�Q�?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	�t{Ic�?�t{Ic�?!�t{Ic�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q3֮���X@y��((��?