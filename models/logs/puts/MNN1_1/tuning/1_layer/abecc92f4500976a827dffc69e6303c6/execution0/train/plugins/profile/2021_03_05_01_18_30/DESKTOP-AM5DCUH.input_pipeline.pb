	)u�8Fz5@)u�8Fz5@!)u�8Fz5@	�'�6%�?�'�6%�?!�'�6%�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6)u�8Fz5@3j�J>3@1rM��΢�?A9��m4��?IøDk� @Y�+��f*�?*	      I@2U
Iterator::Model::ParallelMapV2L7�A`�?!     �@@)L7�A`�?1     �@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!gfffff7@)Έ����?1������2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�+e�X�?!������6@)� �	�?1������.@:Preprocessing2F
Iterator::Model�+e�X�?!������F@) �o_�y?1333333)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!������@)���_vOn?1������@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!433333K@)�~j�t�h?1      @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!333333@)a2U0*�c?1333333@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!      9@)/n��R?1������@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�'�6%�?I�`Ƭ_�X@Q��3��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	3j�J>3@3j�J>3@!3j�J>3@      ��!       "	rM��΢�?rM��΢�?!rM��΢�?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	øDk� @øDk� @!øDk� @B      ��!       J	�+��f*�?�+��f*�?!�+��f*�?R      ��!       Z	�+��f*�?�+��f*�?!�+��f*�?b      ��!       JGPUY�'�6%�?b q�`Ƭ_�X@y��3��?