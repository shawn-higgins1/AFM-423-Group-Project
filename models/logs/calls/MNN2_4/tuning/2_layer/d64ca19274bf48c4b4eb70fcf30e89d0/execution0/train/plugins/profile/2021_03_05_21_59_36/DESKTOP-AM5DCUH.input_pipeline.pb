	��s|��7@��s|��7@!��s|��7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��s|��7@:\�=�5@1���oa��?A�3��7�?I�����9@*	     �G@2U
Iterator::Model::ParallelMapV2�!��u��?!      >@)�!��u��?1      >@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM�St$�?!����
8@)Έ����?1W�+��3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM�St$�?!����
8@)��H�}}?1��F}g�.@:Preprocessing2F
Iterator::Model��ZӼ�?!G}g���E@) �o_�y?1�����*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!��
br!@)	�^)�p?1��
br!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�]K�=�?!���\ALL@)Ǻ���f?1�٨�l�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!��
br@)����Mb`?1��
br@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!r1���:@)a2U0*�S?1Q�٨�l@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIG�-ң�X@QMn�t��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	:\�=�5@:\�=�5@!:\�=�5@      ��!       "	���oa��?���oa��?!���oa��?*      ��!       2	�3��7�?�3��7�?!�3��7�?:	�����9@�����9@!�����9@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qG�-ң�X@yMn�t��?