	�"�dT�5@�"�dT�5@!�"�dT�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�"�dT�5@�!���=3@1ePmp"��?A�H.�!��?I£�#�B @*	23333sH@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateΈ����?!+�/�C@)������?1\w��|A@:Preprocessing2U
Iterator::Model::ParallelMapV2�
F%u�?!#����9@)�
F%u�?1#����9@:Preprocessing2F
Iterator::Model���<,�?!�Pvn�$D@)y�&1�|?1V�nL>�,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t�x?!��f5�(@)����Mbp?1��c�#\ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�<,Ԛ�?!`���Z�M@)F%u�k?1�bK�m�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!��c�#\@)����Mb`?1��c�#\@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap8��d�`�?!���y�XD@)��_�LU?1�έ�D@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!ɀz�r�?)��H�}M?1ɀz�r�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�0�Qġ�?)a2U0*�C?1�0�Qġ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�4��3�X@Q��2��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�!���=3@�!���=3@!�!���=3@      ��!       "	ePmp"��?ePmp"��?!ePmp"��?*      ��!       2	�H.�!��?�H.�!��?!�H.�!��?:	£�#�B @£�#�B @!£�#�B @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�4��3�X@y��2��?