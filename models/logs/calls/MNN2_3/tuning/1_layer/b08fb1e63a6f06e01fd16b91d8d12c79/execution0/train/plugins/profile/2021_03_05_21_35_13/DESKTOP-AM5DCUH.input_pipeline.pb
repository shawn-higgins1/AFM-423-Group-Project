	�	�Y�5@�	�Y�5@!�	�Y�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�	�Y�5@�wJ'3@1��lY�.�?A��ʡE�?I�_u�H��?*	������G@2U
Iterator::Model::ParallelMapV2S�!�uq�?!����c<@)S�!�uq�?1����c<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!��}ylE:@)��_�L�?1��/��6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<�R�?!A�I�7@)_�Q�{?1�}ylE�,@:Preprocessing2F
Iterator::Model��ׁsF�?!4��}�D@)-C��6z?1'u_+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!W'u_!@)	�^)�p?1W'u_!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?�߾�?!��c+�M@)�~j�t�h?14��}yl@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!"h8���@)����Mb`?1"h8���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�~j�t��?!4��}yl9@)/n��R?1�؊��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI}�Z��X@Q� F)���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�wJ'3@�wJ'3@!�wJ'3@      ��!       "	��lY�.�?��lY�.�?!��lY�.�?*      ��!       2	��ʡE�?��ʡE�?!��ʡE�?:	�_u�H��?�_u�H��?!�_u�H��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q}�Z��X@y� F)���?