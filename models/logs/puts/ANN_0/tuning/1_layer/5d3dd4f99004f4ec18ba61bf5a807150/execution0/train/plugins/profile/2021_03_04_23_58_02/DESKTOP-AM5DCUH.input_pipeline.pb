	ٖg)�2@ٖg)�2@!ٖg)�2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ٖg)�2@��Ia��0@1˝�`8��?A�f��j+�?Ih+�m�?*	233333G@2U
Iterator::Model::ParallelMapV2�HP��?!����K:@)�HP��?1����K:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!+�4�rO=@)������?1      9@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!,�4�r7@)lxz�,C|?1���˽-@:Preprocessing2F
Iterator::Model��~j�t�?!�{ayD@)_�Q�{?1+�4�rO-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!o?!��{a @)ŏ1w-!o?1��{a @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?�߾�?!�i�垆M@)a��+ei?1�{a�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!����=@)����Mb`?1����=@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI3�[��X@Q�fy%R��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Ia��0@��Ia��0@!��Ia��0@      ��!       "	˝�`8��?˝�`8��?!˝�`8��?*      ��!       2	�f��j+�?�f��j+�?!�f��j+�?:	h+�m�?h+�m�?!h+�m�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q3�[��X@y�fy%R��?