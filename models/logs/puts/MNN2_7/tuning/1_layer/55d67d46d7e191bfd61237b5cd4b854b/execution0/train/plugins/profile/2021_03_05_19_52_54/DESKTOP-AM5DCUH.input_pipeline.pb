	�>�5@�>�5@!�>�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�>�5@oJy��F3@1���ם�?A�{�Pk�?I�f�R@:@*������F@)       =2U
Iterator::Model::ParallelMapV2tF��_�?!��,��:@)tF��_�?1��,��:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!�Gp�:@)M�O��?1c:��,&6@:Preprocessing2F
Iterator::Model�l����?!��>��HD@)F%u�{?1)�����,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ZӼ�?!�5��P^6@)�~j�t�x?1�5��P*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!����k"@)�J�4q?1����k"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��6��?!p�}�M@)-C��6j?1�}�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!����k@)�J�4a?1����k@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!���k�9@)�~j�t�X?1�5��P
@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI&���V�X@Q?m'��T�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	oJy��F3@oJy��F3@!oJy��F3@      ��!       "	���ם�?���ם�?!���ם�?*      ��!       2	�{�Pk�?�{�Pk�?!�{�Pk�?:	�f�R@:@�f�R@:@!�f�R@:@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q&���V�X@y?m'��T�?