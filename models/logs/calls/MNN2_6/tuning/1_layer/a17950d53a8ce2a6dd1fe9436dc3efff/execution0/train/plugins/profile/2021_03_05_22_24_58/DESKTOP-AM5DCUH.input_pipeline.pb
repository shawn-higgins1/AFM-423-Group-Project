	'l?��1@'l?��1@!'l?��1@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-'l?��1@�k�)\0@1I,)w���?A��ʡE�?I[|
���?*	�����YO@2U
Iterator::Model::ParallelMapV2��_vO�?!���9A@)��_vO�?1���9A@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�l����?!t���m�=@)���Q��?1����f�7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��@��ǈ?!�t�YL3@)��~j�t�?1N6�d�M.@:Preprocessing2F
Iterator::ModelO��e�c�?!q���F@)�ZӼ�}?1���cĥ&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!͓��T@)y�&1�l?1͓��T@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�� �rh�?!�V$�K@)_�Q�k?1Srx�ʰ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!{f��@�@)��_�Le?1{f��@�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�N\�d�X@Q���Q���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�k�)\0@�k�)\0@!�k�)\0@      ��!       "	I,)w���?I,)w���?!I,)w���?*      ��!       2	��ʡE�?��ʡE�?!��ʡE�?:	[|
���?[|
���?![|
���?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�N\�d�X@y���Q���?