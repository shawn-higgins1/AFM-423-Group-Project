	�)���7@�)���7@!�)���7@	\$��0�?\$��0�?!\$��0�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�)���7@�[�v�4@1ڎ����?A
ףp=
�?I�V�f@Y��.���?*	333333I@2U
Iterator::Model::ParallelMapV2��<,Ԋ?!��}���9@)��<,Ԋ?1��}���9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!,˲,�29@)�g��s��?1� � 5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate-C��6�?!Y�eY�e9@)�� �rh�?1�u]�u�0@:Preprocessing2F
Iterator::Model�0�*��?!�0��C@)y�&1�|?1r�q�+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"��u��q?!AA!@)"��u��q?1AA!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�sF���?!�<��<N@)_�Q�k?1������@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!������@)�J�4a?1������@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ZӼ��?!˲,˲,<@)Ǻ���V?1��8��8@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9\$��0�?I ���{X@Q{���7��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�[�v�4@�[�v�4@!�[�v�4@      ��!       "	ڎ����?ڎ����?!ڎ����?*      ��!       2	
ףp=
�?
ףp=
�?!
ףp=
�?:	�V�f@�V�f@!�V�f@B      ��!       J	��.���?��.���?!��.���?R      ��!       Z	��.���?��.���?!��.���?b      ��!       JGPUY\$��0�?b q ���{X@y{���7��?