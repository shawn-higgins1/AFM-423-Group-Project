	g)YNB�7@g)YNB�7@!g)YNB�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-g)YNB�7@�nJy��4@1��H�H�?A�+e�X�?Ig�+��r@*	������I@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateM�O��?!     �C@)�&S��?1     �A@:Preprocessing2U
Iterator::Model::ParallelMapV2��<,Ԋ?!     �9@)��<,Ԋ?1     �9@:Preprocessing2F
Iterator::Model��ׁsF�?!     VC@)S�!�uq{?1     ,*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq{?!     ,*@)/n��r?1     0!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2U0*��?!     �N@)-C��6j?1      @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!     �@)HP�s�b?1     �@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj�t��?!     �D@)��_�LU?1     P@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��_�LU?!     P@)��_�LU?1     P@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!     ��?)Ǻ���F?1     ��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�d����X@Q7�f���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�nJy��4@�nJy��4@!�nJy��4@      ��!       "	��H�H�?��H�H�?!��H�H�?*      ��!       2	�+e�X�?�+e�X�?!�+e�X�?:	g�+��r@g�+��r@!g�+��r@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�d����X@y7�f���?