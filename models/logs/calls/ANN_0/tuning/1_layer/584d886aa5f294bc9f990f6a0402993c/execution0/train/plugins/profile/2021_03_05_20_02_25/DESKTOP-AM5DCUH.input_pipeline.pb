	�1��(3@�1��(3@!�1��(3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�1��(3@�$�)��1@1o+�6+�?A��@�m�?I�xy:W�?*	gffff�M@2U
Iterator::Model::ParallelMapV2?�ܵ�|�?!�K�k&;@)?�ܵ�|�?1�K�k&;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapr�����?!}��%�=@)�!��u��?1���7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-!�?!uR��9@) �o_Ή?1L���u?5@:Preprocessing2F
Iterator::ModelM�St$�?!��V?C@)9��v��z?1�XC$�%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip'�Wʢ?!H:����N@)ŏ1w-!o?1uR��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��H�}m?!|����H@)��H�}m?1|����H@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!��F���@)��_�Le?1��F���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIG����X@Q�ܜ�3��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�$�)��1@�$�)��1@!�$�)��1@      ��!       "	o+�6+�?o+�6+�?!o+�6+�?*      ��!       2	��@�m�?��@�m�?!��@�m�?:	�xy:W�?�xy:W�?!�xy:W�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qG����X@y�ܜ�3��?