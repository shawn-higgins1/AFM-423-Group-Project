	�ri��+7@�ri��+7@!�ri��+7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�ri��+7@Ʀ�B �4@1�^zo�?AEGr��?I,)w���@*	�����LJ@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate;�O��n�?!��-g:A@)�Pk�w�?1�z��m:@:Preprocessing2U
Iterator::Model::ParallelMapV2��0�*�?!ּR=�n6@)��0�*�?1ּR=�n6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t��?!X���M�6@)M�O��?1
��ߖ33@:Preprocessing2F
Iterator::Model�&S��?!�6���LA@)-C��6z?1]`;e�U(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice	�^)�p?!x���-@)	�^)�p?1x���-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip"��u���?!��$�YP@)a��+ei?1Z�"��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!n�6���@)ŏ1w-!_?1n�6���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�N@aÓ?!��Q��XB@)��_�LU?1L>@Ҙ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���X@Q�����P�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Ʀ�B �4@Ʀ�B �4@!Ʀ�B �4@      ��!       "	�^zo�?�^zo�?!�^zo�?*      ��!       2	EGr��?EGr��?!EGr��?:	,)w���@,)w���@!,)w���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���X@y�����P�?