	{�V���1@{�V���1@!{�V���1@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-{�V���1@�X�E�E0@1���ZӼ�?Akc섗�?I���f��?*	     @J@2U
Iterator::Model::ParallelMapV2�<,Ԛ�?!�<��<�;@)�<,Ԛ�?1�<��<�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty�&1��?!������:@)��@��ǈ?1�0�07@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u��?!J�$I�$9@)��~j�t�?1�a�a2@:Preprocessing2F
Iterator::Model�I+��?!=��<��D@)���_vO~?1�0�0,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!�0�0@)���_vOn?1�0�0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Zd;�?!�0�0M@)F%u�k?1J�$I�$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!=��<��@)ŏ1w-!_?1=��<��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noId[F�X@Q�M~@���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�X�E�E0@�X�E�E0@!�X�E�E0@      ��!       "	���ZӼ�?���ZӼ�?!���ZӼ�?*      ��!       2	kc섗�?kc섗�?!kc섗�?:	���f��?���f��?!���f��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qd[F�X@y�M~@���?