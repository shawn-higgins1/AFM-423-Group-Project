	��w�Gk8@��w�Gk8@!��w�Gk8@	U�E��j�?U�E��j�?!U�E��j�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��w�Gk8@����5@1r��Q���?A��y�)�?I�rh��|@Y~�k�,	p?*	33333sF@2U
Iterator::Model::ParallelMapV2������?!�r���9@)������?1�r���9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;�O��?!�p��ʜ9@)�j+��݃?1���"��5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'�����?!|(���7@)�ZӼ�}?1F�&�6�/@:Preprocessing2F
Iterator::Model�:pΈ�?!#��1�'D@)9��v��z?1�؄��,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!���[	 @)��H�}m?1���[	 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!�|e��M@)-C��6j?1�����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!���[	@)��H�}]?1���[	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��@��ǈ?!2g���:@)Ǻ���V?1��� ��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9U�E��j�?I�����X@Q!CY+�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����5@����5@!����5@      ��!       "	r��Q���?r��Q���?!r��Q���?*      ��!       2	��y�)�?��y�)�?!��y�)�?:	�rh��|@�rh��|@!�rh��|@B      ��!       J	~�k�,	p?~�k�,	p?!~�k�,	p?R      ��!       Z	~�k�,	p?~�k�,	p?!~�k�,	p?b      ��!       JGPUYU�E��j�?b q�����X@y!CY+�?