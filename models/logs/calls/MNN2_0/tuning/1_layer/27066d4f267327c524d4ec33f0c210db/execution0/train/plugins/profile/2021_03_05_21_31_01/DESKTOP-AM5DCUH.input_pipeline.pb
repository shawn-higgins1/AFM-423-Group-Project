	����z6@����z6@!����z6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����z6@:�w�f3@1�W�2�?A9��m4��?I���K�:@*	�����K@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateM�O��?!���z�B@)HP�sג?1�W-0c�@@:Preprocessing2U
Iterator::Model::ParallelMapV2K�=�U�?!s�y�:<@)K�=�U�?1s�y�:<@:Preprocessing2F
Iterator::Model䃞ͪϕ?!�: B�C@)�~j�t�x?1_����#&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat"��u���?!�Э8��/@)�~j�t�x?1_����#&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip*��Dؠ?!n��߽YN@)a��+ei?1U����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!�6�W-0@)��_�Le?1�6�W-0@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensora2U0*�S?!�f=Q�@)a2U0*�S?1�f=Q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�g��s��?!�w^��C@)����MbP?1�� 2��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�f=Q��?)a2U0*�C?1�f=Q��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�O۰��X@Qf?X�'��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	:�w�f3@:�w�f3@!:�w�f3@      ��!       "	�W�2�?�W�2�?!�W�2�?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	���K�:@���K�:@!���K�:@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�O۰��X@yf?X�'��?