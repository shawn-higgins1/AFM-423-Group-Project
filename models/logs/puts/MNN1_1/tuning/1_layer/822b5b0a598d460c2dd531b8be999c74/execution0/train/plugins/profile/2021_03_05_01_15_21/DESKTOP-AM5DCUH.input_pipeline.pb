	�j���?6@�j���?6@!�j���?6@	���1&��?���1&��?!���1&��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�j���?6@��
}��3@1|�(B�v�?A�H.�!��?I�J�*�@Ys��c�Ȱ?*	fffff&M@2U
Iterator::Model::ParallelMapV2��d�`T�?!��;��>@)��d�`T�?1��;��>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�St$���?!S���1y<@)�
F%u�?1UI_!��5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!'�gL�4@)M�O��?1���R1@:Preprocessing2F
Iterator::ModelV}��b�?!>u�6�ZE@)y�&1�|?1�dK�(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!�K���@)�q����o?1�K���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�o_��?!��L@)-C��6j?1�7�K��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!2y���@)�J�4a?12y���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;�O��n�?!l�����>@)Ǻ���V?1�Px�6@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���1&��?It�]o��X@QW_8���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��
}��3@��
}��3@!��
}��3@      ��!       "	|�(B�v�?|�(B�v�?!|�(B�v�?*      ��!       2	�H.�!��?�H.�!��?!�H.�!��?:	�J�*�@�J�*�@!�J�*�@B      ��!       J	s��c�Ȱ?s��c�Ȱ?!s��c�Ȱ?R      ��!       Z	s��c�Ȱ?s��c�Ȱ?!s��c�Ȱ?b      ��!       JGPUY���1&��?b qt�]o��X@yW_8���?