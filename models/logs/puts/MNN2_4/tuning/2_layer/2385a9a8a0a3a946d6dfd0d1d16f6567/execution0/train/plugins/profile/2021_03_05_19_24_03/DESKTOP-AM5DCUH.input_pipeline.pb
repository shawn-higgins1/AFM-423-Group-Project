	��je7@��je7@!��je7@	d��.�կ?d��.�կ?!d��.�կ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��je7@C;�Y�i4@1������?A �o_Ω?I�s(CU�@Y�_>Y1\�?*	      H@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!UUUUUU<@)��0�*�?1UUUUU�8@:Preprocessing2U
Iterator::Model::ParallelMapV2������?!�����*8@)������?1�����*8@:Preprocessing2F
Iterator::ModelΈ����?!     `C@)y�&1�|?1�����*-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�0�*�?!UUUUUu5@)9��v��z?1UUUUU+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!������@)ŏ1w-!o?1������@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%u��?!    �N@)���_vOn?1VUUUU�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��@��ǈ?!UUUUU59@)��H�}]?1      @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!      @)��H�}]?1      @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9e��.�կ?I�-�X@Q��Ʌ}�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	C;�Y�i4@C;�Y�i4@!C;�Y�i4@      ��!       "	������?������?!������?*      ��!       2	 �o_Ω? �o_Ω?! �o_Ω?:	�s(CU�@�s(CU�@!�s(CU�@B      ��!       J	�_>Y1\�?�_>Y1\�?!�_>Y1\�?R      ��!       Z	�_>Y1\�?�_>Y1\�?!�_>Y1\�?b      ��!       JGPUYe��.�կ?b q�-�X@y��Ʌ}�?