	81$'�7@81$'�7@!81$'�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-81$'�7@���5@1�-����?A�c]�F�?I�|A�@*	������G@2U
Iterator::Model::ParallelMapV2�{�Pk�?![4�T;@)�{�Pk�?1[4�T;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!R���;@)46<�R�?1A�I�7@:Preprocessing2F
Iterator::Model�N@aÓ?!�pR��qD@)-C��6z?1'u_+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ZӼ�?!`[4�5@)�HP�x?1h8����)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!W'u_!@)	�^)�p?1W'u_!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6�;Nё�?!/���M@)-C��6j?1'u_@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�؊��@)/n��b?1�؊��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!�c+���8@)Ǻ���V?1d+����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI �ڲ��X@Q@GI��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���5@���5@!���5@      ��!       "	�-����?�-����?!�-����?*      ��!       2	�c]�F�?�c]�F�?!�c]�F�?:	�|A�@�|A�@!�|A�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q �ڲ��X@y@GI��?