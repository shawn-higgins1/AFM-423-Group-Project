	��e1�-5@��e1�-5@!��e1�-5@	) �C��?) �C��?!) �C��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��e1�-5@�?����2@1�yUg��?A��@��Ǩ?IQ���J @Ya��+ey?*fffff�H@)       =2U
Iterator::Model::ParallelMapV2F%u��?!0�n\�:@)F%u��?10�n\�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!��e��8@)�0�*�?1O�3�z�4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�(��0�?!��e��8@)"��u���?1+�<W�q1@:Preprocessing2F
Iterator::Model�N@aÓ?!2 ���C@)�HP�x?1g��]�(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���Q��?!��u�mN@)����Mbp?1Lfǀ(: @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!��T1@)���_vOn?1��T1@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!Lfǀ(:@)����Mb`?1Lfǀ(:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�?�߾�?!�E"���;@)Ǻ���V?1���Mҷ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9) �C��?IJ��ɥX@Q��c���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�?����2@�?����2@!�?����2@      ��!       "	�yUg��?�yUg��?!�yUg��?*      ��!       2	��@��Ǩ?��@��Ǩ?!��@��Ǩ?:	Q���J @Q���J @!Q���J @B      ��!       J	a��+ey?a��+ey?!a��+ey?R      ��!       Z	a��+ey?a��+ey?!a��+ey?b      ��!       JGPUY) �C��?b qJ��ɥX@y��c���?