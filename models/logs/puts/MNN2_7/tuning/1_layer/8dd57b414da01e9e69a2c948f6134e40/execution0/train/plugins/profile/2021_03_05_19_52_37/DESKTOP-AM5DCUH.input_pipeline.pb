	ٙB�5~5@ٙB�5~5@!ٙB�5~5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ٙB�5~5@~r 
�2@1�����?Ax��-�?I�z�h@*	gfffffI@2U
Iterator::Model::ParallelMapV2�5�;Nё?!�@ A@)�5�;Nё?1�@ A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!��f��l7@)��ׁsF�?1�����|3@:Preprocessing2F
Iterator::Model�~j�t��?!���|>�G@)F%u�{?1~�����)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�g��s��?!���v��4@)9��v��z?1�r�\.�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!�D"�H$ @)	�^)�p?1�D"�H$ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!��`J@)a��+ei?1�F��h@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�����~@)����Mb`?1�����~@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!O���t:7@)a2U0*�S?1�\.���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIk��co�X@Q��NH�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~r 
�2@~r 
�2@!~r 
�2@      ��!       "	�����?�����?!�����?*      ��!       2	x��-�?x��-�?!x��-�?:	�z�h@�z�h@!�z�h@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qk��co�X@y��NH�?