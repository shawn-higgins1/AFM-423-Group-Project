	��l a4@��l a4@!��l a4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��l a4@�I�U�2@1w;S��?A6=((E+�?Ipy���?*	43333�G@2U
Iterator::Model::ParallelMapV2���S㥋?!�D�#{<@)���S㥋?1�D�#{<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF%u��?!�g *�;@)/�$��?1M����&6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA��ǘ��?!��&��j7@)�ZӼ�}?1@9��2�-@:Preprocessing2F
Iterator::Model8��d�`�?!�>���D@)-C��6z?1����+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!l��Ӭ� @)����Mbp?1l��Ӭ� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)\���(�?!Z��)M@)-C��6j?1����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_vOf?!�=Q��@)��_vOf?1�=Q��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��s&��X@Q��cv��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�I�U�2@�I�U�2@!�I�U�2@      ��!       "	w;S��?w;S��?!w;S��?*      ��!       2	6=((E+�?6=((E+�?!6=((E+�?:	py���?py���?!py���?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��s&��X@y��cv��?